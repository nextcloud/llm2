# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
The main module of the llm2 app
"""
import asyncio
import os
import logging
import traceback
from contextlib import asynccontextmanager
from json import JSONDecodeError
from threading import Event, Thread
from time import perf_counter, sleep, strftime

from niquests import RequestException
from task_processors import generate_task_processors
from fastapi import FastAPI
from nc_py_api import AsyncNextcloudApp, NextcloudApp, NextcloudException
from nc_py_api.ex_app import LogLvl, persistent_storage, run_app, set_handlers
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider, ShapeEnumValue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log(nc, level, content):
    logger.log((level+1)*10, content)
    if level < LogLvl.WARNING:
        return
    try:
        asyncio.run(nc.log(level, content))
    except:
        pass

models_to_fetch = {
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/4f0c246f125fc7594238ebe7beb1435a8335f519/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": { "save_path": os.path.join(persistent_storage(), "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf") },
    "https://huggingface.co/unsloth/Olmo-3-7B-Instruct-GGUF/resolve/main/Olmo-3-7B-Instruct-Q4_K_M.gguf": { "save_path": os.path.join(persistent_storage(), "Olmo-3-7B-Instruct-Q4_K_M.gguf") },
}
app_enabled = Event()
trigger = Event()
CHECK_INTERVAL = 5
CHECK_INTERVAL_WITH_TRIGGER = 5 * 60
CHECK_INTERVAL_ON_ERROR = 10

@asynccontextmanager
async def lifespan(_app: FastAPI):
    set_handlers(
        APP,
        enabled_handler, # type: ignore
        models_to_fetch=models_to_fetch,
        trigger_handler=trigger_handler
    )
    nc = NextcloudApp()
    if nc.enabled_state:
        app_enabled.set()
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)


def background_thread_task():
    nc = NextcloudApp()

    while not app_enabled.is_set():
        sleep(5)

    provider_ids = set()
    task_type_ids = set()
    task_processors = generate_task_processors()
    for task_processor_name, _ in task_processors.items():
        provider_ids.add("llm2:" + task_processor_name)
        (model, task) = task_processor_name.split(":", 1)
        task_type_ids.add(task)

    while True:
        if not app_enabled.is_set():
            sleep(30)
            continue

        current_minute = int(strftime("%M"))
        if current_minute % 5 == 0:
            # scan dir and load new models every 5 minutes
            provider_ids = set()
            task_type_ids = set()
            task_processors = generate_task_processors(task_processors)
            for task_processor_name, _ in task_processors.items():
                provider_ids.add("llm2:" + task_processor_name)
                (model, task) = task_processor_name.split(":", 1)
                task_type_ids.add(task)

        try:
            response = nc.providers.task_processing.next_task(list(provider_ids), list(task_type_ids))
            if not response:
                wait_for_tasks()
                continue
        except (NextcloudException, RequestException, JSONDecodeError) as e:
            log(nc, LogLvl.ERROR, f"Network error fetching the next task {e}")
            wait_for_tasks(CHECK_INTERVAL_ON_ERROR)
            continue

        task = response["task"]
        provider = response["provider"]

        try:
            task_processor_name = provider["name"][5:]
            log(nc, LogLvl.INFO, f"chain: {task_processor_name}")
            task_processor_loader = task_processors.get(task_processor_name)
            if task_processor_loader is None:
                NextcloudApp().providers.task_processing.report_result(
                    task["id"], error_message="Requested model is not available"
                )
                continue
            task_processor = task_processor_loader()
            log(nc, LogLvl.INFO, "Generating reply")
            time_start = perf_counter()
            log(nc, LogLvl.INFO, task.get("input"))
            result = task_processor(task.get("input"))
            log(nc, LogLvl.INFO, f"reply generated: {round(float(perf_counter() - time_start), 2)}s")
            log(nc, LogLvl.INFO, result)
            nc.providers.task_processing.report_result(
                task["id"],
                result,
            )
        except (NextcloudException, RequestException, JSONDecodeError) as e:
            # Error when reporting the result
            exception_info = traceback.format_exception(type(e), e, e.__traceback__)
            log(nc, LogLvl.ERROR, f"Error: {''.join(exception_info)}")
        except Exception as e:  # noqa
            exception_info = traceback.format_exception(type(e), e, e.__traceback__)
            log(nc, LogLvl.ERROR, f"Error: {''.join(exception_info)}")
            try:
                log(nc, LogLvl.ERROR, str(e))
                nc.providers.task_processing.report_result(task["id"], error_message=str(e))
            except (NextcloudException, RequestException) as net_err:
                log(nc, LogLvl.INFO, f"Network error in reporting the error: {net_err}")


def start_bg_task():
    t = Thread(target=background_thread_task, args=())
    t.start()


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    global app_enabled
    log(nc, LogLvl.INFO, f"enabled={enabled}")

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
                     input_shape_enum_values= {
                            "tone": [
                                ShapeEnumValue(name= "Friendlier", value= "friendlier"),
                                ShapeEnumValue(name= "More formal", value= "more formal"),
                                ShapeEnumValue(name= "Funnier", value= "funnier"),
                                ShapeEnumValue(name= "More casual", value= "more casual"),
                                ShapeEnumValue(name= "More urgent", value= "more urgent"),
                            ],
                        } if task == "core:text2text:changetone" else {}
                )
                await nc.providers.task_processing.register(provider)
                log(nc, LogLvl.INFO, f"Registered {task_processor_name}")
                app_enabled.set()
            except Exception as e:
                log(nc, LogLvl.ERROR, f"Failed to register {model} - {task}, Error: {e}\n")
                break
    else:
        app_enabled.clear()
        for task_processor_name in task_processors:
            try:
                await nc.providers.task_processing.unregister("llm2:" + task_processor_name)
                log(nc, LogLvl.INFO, f"Unregistered {task_processor_name}")
            except Exception as e:
                log(nc, LogLvl.ERROR, f"Failed to unregister {task_processor_name}, Error: {e}\n")
                break

    return ""


def trigger_handler(providerId: str):
    print('TRIGGER called')
    trigger.set()

def wait_for_tasks(interval = None):
    global CHECK_INTERVAL
    global CHECK_INTERVAL_WITH_TRIGGER
    actual_interval = CHECK_INTERVAL if interval is None else interval
    if trigger.wait(timeout=actual_interval):
        CHECK_INTERVAL = CHECK_INTERVAL_WITH_TRIGGER
    trigger.clear()


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
