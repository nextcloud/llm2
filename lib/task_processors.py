# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registers all task processors based on the models/ directory contents
"""

import json
import os
from functools import cache

from langchain_community.chat_models import ChatLlamaCpp

from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
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

dir_path = os.path.dirname(os.path.realpath(__file__))
models_folder_path = os.path.join(dir_path , "../models/")

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


@cache
def generate_llm(file_name):
    model_config = get_model_config(file_name)

    path = os.path.join(models_folder_path, file_name)

    if not os.path.exists(path):
        path = os.path.join(persistent_storage(), file_name)

    compute_device = os.getenv("COMPUTE_DEVICE", "CUDA")
    try:
        llm = LlamaCpp(
            model_path=path,
            **{
                "n_gpu_layers": (0, -1)[compute_device != "CPU"],
                **model_config["loader_config"],
            },
        )
    except Exception as e:
        print(f"Failed to load model '{path}' with compute device '{compute_device}'")
        raise e

    return llm

@cache
def generate_llm_chain(file_name):
    model_config = get_model_config(file_name)
    print(model_config)
    llm = generate_llm(file_name)

    prompt = PromptTemplate.from_template(model_config['prompt'])

    return prompt | llm

@cache
def generate_chat_chain(file_name):
    model_config = get_model_config(file_name)

    path = os.path.join(models_folder_path, file_name)

    if not os.path.exists(path):
        path = os.path.join(persistent_storage(), file_name)

    compute_device = os.getenv("COMPUTE_DEVICE", "CUDA")
    try:
        llm = ChatLlamaCpp(
            model_path=path,
            n_batch=1,
            **{
                "n_gpu_layers": (0, -1)[compute_device != "CPU"],
                **model_config["loader_config"],
            },
        )
    except Exception as e:
        print(f"Failed to load model '{path}' with compute device '{compute_device}'")
        raise e

    return llm


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

    task_processors[model_name + ":core:text2text:summary"] = lambda: SummarizeProcessor(generate_chat_chain(file_name), n_ctx)
    task_processors[model_name + ":core:text2text:headline"] = lambda: HeadlineProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text:topics"] = lambda: TopicsProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text:simplification"] = lambda: SimplifyProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text:reformulation"] = lambda: ReformulateProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:contextwrite"] = lambda: ContextWriteProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text"] = lambda: FreePromptProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text:chat"] = lambda: ChatProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text:proofread"] = lambda: ProofreadProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text:changetone"] = lambda: ChangeToneProcessor(generate_chat_chain(file_name))
    task_processors[model_name + ":core:text2text:chatwithtools"] = lambda: ChatWithToolsProcessor(generate_chat_chain(file_name))
    
    # chains[model_name + ":core:contextwrite"] = lambda: ContextWriteChain(llm_chain=llm_chain())
