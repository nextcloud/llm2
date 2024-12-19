# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
The main module of the llm2 app
"""

import os
from contextlib import asynccontextmanager
from json import JSONDecodeError
from threading import Event, Thread
from time import perf_counter, sleep

import httpx
from task_processors import generate_task_processors
from fastapi import FastAPI
from nc_py_api import AsyncNextcloudApp, NextcloudApp, NextcloudException
from nc_py_api.ex_app import LogLvl, persistent_storage, run_app, set_handlers
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider

models_to_fetch = {
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/4f0c246f125fc7594238ebe7beb1435a8335f519/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": { "save_path": os.path.join(persistent_storage(), "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf") },
}
app_enabled = Event()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    set_handlers(
        APP,
        enabled_handler, # type: ignore
        #models_to_fetch=models_to_fetch,
    )
    nc = NextcloudApp()
    if nc.enabled_state:
        app_enabled.set()
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)


def background_thread_task(task_processors: dict):
    nc = NextcloudApp()

    provider_ids = set()
    task_type_ids = set()
    for task_processor_name, _ in task_processors.items():
        provider_ids.add("llm2:" + task_processor_name)
        (model, task) = task_processor_name.split(":", 1)
        task_type_ids.add(task)

    while True:
        if not app_enabled.is_set():
            sleep(30)
            continue

        try:
            response = nc.providers.task_processing.next_task(list(provider_ids), list(task_type_ids))
            if not response:
                sleep(5)
                continue
        except (NextcloudException, httpx.RequestError, JSONDecodeError) as e:
            print("Network error fetching the next task", e, flush=True)
            sleep(5)
            continue

        task = response["task"]
        provider = response["provider"]

        try:
            task_processor_name = provider["name"][5:]
            print(f"chain: {task_processor_name}", flush=True)
            task_processor_loader = task_processors.get(task_processor_name)
            if task_processor_loader is None:
                NextcloudApp().providers.task_processing.report_result(
                    task["id"], error_message="Requested model is not available"
                )
                continue
            task_processor = task_processor_loader()
            print("Generating reply", flush=True)
            time_start = perf_counter()
            print(task.get("input"), flush=True)
            result = task_processor(task.get("input"))
            print(f"reply generated: {round(float(perf_counter() - time_start), 2)}s", flush=True)
            print(result, flush=True)
            nc.providers.task_processing.report_result(
                task["id"],
                result,
            )
        except (NextcloudException, httpx.RequestError, JSONDecodeError) as e:
            print("Network error:", e, flush=True)
            sleep(5)
        except Exception as e:  # noqa
            print("Error:", e, flush=True)
            try:
                nc.log(LogLvl.ERROR, str(e))
                nc.providers.task_processing.report_result(task["id"], error_message=str(e))
            except (NextcloudException, httpx.RequestError) as net_err:
                print("Network error in reporting the error:", net_err, flush=True)

            sleep(5)


def start_bg_task():
    task_processors = generate_task_processors()
    t = Thread(target=background_thread_task, args=(task_processors,))
    t.start()


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    global app_enabled
    print(f"enabled={enabled}", flush=True)

    task_processors = generate_task_processors()

    if enabled is True:
        for task_processor_name in task_processors:
            (model, task) = task_processor_name.split(":", 1)
            try:
                provider = TaskProcessingProvider(
                    id="llm2:" + task_processor_name,
                    name="Local Large language Model: " + model,
                    task_type=task,
                    expected_runtime=30,
                )
                await nc.providers.task_processing.register(provider)
                print(f"Registered {task_processor_name}", flush=True)
                app_enabled.set()
            except Exception as e:
                print(f"Failed to register {model} - {task}, Error: {e}\n", flush=True)
                break
    else:
        app_enabled.clear()
        for task_processor_name in task_processors:
            try:
                await nc.providers.task_processing.unregister("llm2:" + task_processor_name)
                print(f"Unregistered {task_processor_name}", flush=True)
            except Exception as e:
                print(f"Failed to unregister {task_processor_name}, Error: {e}\n", flush=True)
                break

    return ""


if __name__ == "__main__":
    # print(os.environ["APP_HOST"], flush=True)
    # print(os.environ["APP_ID"], flush=True)
    # print(os.environ["APP_PORT"], flush=True)
    # print(os.environ["APP_SECRET"], flush=True)
    # print(os.environ["APP_VERSION"], flush=True)
    # print(os.environ["COMPUTE_DEVICE"], flush=True)
    # print(os.environ["NEXTCLOUD_URL"], flush=True)
    # print(os.environ["APP_PERSISTENT_STORAGE"], flush=True)
    run_app("main:APP", log_level="trace")
