# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registers all task processors based on the models/ directory contents
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections import deque
from functools import cache
from threading import Thread

import niquests
from langchain_openai import ChatOpenAI
from nc_py_api.ex_app import persistent_storage

from chat import ChatProcessor
from free_prompt import FreePromptProcessor
from headline import HeadlineProcessor
from contextwrite import ContextWriteProcessor
from reformulate import ReformulateProcessor
from simplify import SimplifyProcessor
from proofread import ProofreadProcessor
from change_tone import ChangeToneProcessor
from chatwithtools import ChatWithToolsProcessor
from topics import TopicsProcessor
from summarize import SummarizeProcessor
from reformat_paragraphs import ReformatParagraphsProcessor

dir_path = os.path.dirname(os.path.realpath(__file__))
models_folder_path = os.path.join(dir_path , "../models/")

# Maps file_name -> (Popen, port) for cleanup on shutdown
_server_processes: dict[str, tuple[subprocess.Popen, int]] = {}

logger = logging.getLogger(__name__)


class _ServerLogPipe:
    """Forward a subprocess pipe to the logger, keeping the last lines for error context."""
    def __init__(self, model_name: str, tail_lines: int = 50) -> None:
        self.prefix = f"[llama-cpp-server:{model_name}] "
        self._tail: deque[str] = deque(maxlen=tail_lines)

    def consume(self, stream) -> None:
        try:
            for line in stream:
                line = line.rstrip()
                self._tail.append(line)
                logger.info(self.prefix + line)
        except Exception:
            pass

    def tail(self) -> str:
        return "\n".join(self._tail)


def get_model_config(file_name):
    file_name = file_name.split('.gguf')[0]
    if os.path.exists(os.path.join(models_folder_path, file_name + ".json")):
        model_config_path = os.path.join(models_folder_path, file_name + ".json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    elif os.path.exists(os.path.join(persistent_storage(), file_name + ".json")):
        model_config_path = os.path.join(persistent_storage(), file_name + ".json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config_path = os.path.join(dir_path, "../default_config", "config.json")
        with open(model_config_path, "r") as f:
            default_config = json.load(f)

        if file_name in default_config:
            model_config = default_config[file_name]
        else:
            model_config = default_config['default']
    return model_config


def get_n_parallel(model_name: str) -> int:
    return get_model_config(model_name)["loader_config"].get("n_parallel", 1)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(proc: subprocess.Popen, port: int, log_pipe: _ServerLogPipe, timeout: float = 300.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"llama-server exited with code {exit_code} before becoming ready. "
                f"Last output:\n{log_pipe.tail()}"
            )
        try:
            resp = niquests.get(url, timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(
        f"llama-server on port {port} did not become ready within {timeout}s. "
        f"Last output:\n{log_pipe.tail()}"
    )


_SERVER_SCRIPT_PATH = os.path.join(dir_path, "llama_server.py")


@cache
def generate_chat_model(file_name: str) -> ChatOpenAI:
    model_config = get_model_config(file_name)
    loader_config = model_config["loader_config"]

    path = os.path.join(models_folder_path, file_name)
    if not os.path.exists(path):
        path = os.path.join(persistent_storage(), file_name)

    compute_device = os.getenv("COMPUTE_DEVICE", "CUDA")
    n_gpu_layers = -1 if compute_device != "CPU" else 0

    port = _find_free_port()
    model_alias = file_name.split(".gguf")[0]

    server_config = json.dumps({
        "model_path": path,
        "hostname": "127.0.0.1",
        "port": port,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": loader_config.get("n_batch", 512),
        "n_parallel": loader_config.get("n_parallel", 1),
        "cont_batching": True,
        **loader_config,
    })

    logger.info(f"Starting llama-server for {file_name} on port {port}")
    try:
        proc = subprocess.Popen(
            [sys.executable, _SERVER_SCRIPT_PATH, server_config],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as e:
        raise RuntimeError(f"Failed to spawn llama-server subprocess for {file_name}: {e}") from e

    log_pipe = _ServerLogPipe(model_alias)
    Thread(target=log_pipe.consume, args=(proc.stdout,), daemon=True).start()
    _server_processes[file_name] = (proc, port)

    try:
        _wait_for_server(proc, port, log_pipe)
    except Exception:
        _server_processes.pop(file_name, None)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise
    logger.info(f"llama-cpp-server for {file_name} ready on port {port}")

    model_kwargs: dict = {}
    if loader_config.get("stop"):
        model_kwargs["stop"] = loader_config["stop"]

    return ChatOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="not-needed",
        model=model_alias,
        max_tokens=loader_config.get("max_tokens", 2048),
        temperature=loader_config.get("temperature", 0.7),
        model_kwargs=model_kwargs,
    )


def stop_all_servers() -> None:
    for file_name, (proc, port) in _server_processes.items():
        logger.info(f"Stopping llama-cpp-server for {file_name} (port {port})")
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception as e:
            logger.warning(f"Error stopping server for {file_name}: {e}")
            try:
                proc.kill()
            except Exception:
                pass
    _server_processes.clear()


def generate_task_processors(task_processors = {}):
    for file in os.scandir(models_folder_path):
        if file.name.endswith(".gguf"):
            if file.name.split('.gguf')[0] in task_processors:
                continue
            generate_task_processors_for_model(file.name, task_processors)

    for file in os.scandir(persistent_storage()):
        if file.name.endswith('.gguf'):
            if file.name.split('.gguf')[0] in task_processors:
                continue
            generate_task_processors_for_model(file.name, task_processors)

    return task_processors


def generate_task_processors_for_model(file_name, task_processors):
    model_name = file_name.split('.gguf')[0]
    n_ctx = get_model_config(file_name)["loader_config"]["n_ctx"]

    task_processors[model_name + ":core:text2text:summary"] = lambda: SummarizeProcessor(generate_chat_model(file_name), n_ctx)
    task_processors[model_name + ":core:text2text:headline"] = lambda: HeadlineProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:topics"] = lambda: TopicsProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:simplification"] = lambda: SimplifyProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:reformulation"] = lambda: ReformulateProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:contextwrite"] = lambda: ContextWriteProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text"] = lambda: FreePromptProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:chat"] = lambda: ChatProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:proofread"] = lambda: ProofreadProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:changetone"] = lambda: ChangeToneProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:chatwithtools"] = lambda: ChatWithToolsProcessor(generate_chat_model(file_name))
    task_processors[model_name + ":core:text2text:reformatparagraphs"] = lambda: ReformatParagraphsProcessor(generate_chat_model(file_name))
