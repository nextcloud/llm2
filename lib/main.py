"""Tha main module of the llm2 app
"""

import os
from contextlib import asynccontextmanager
from threading import Event, Thread
from time import perf_counter, sleep

import httpx
from chains import generate_chains
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
        models_to_fetch=models_to_fetch,
    )
    try:
        nc = NextcloudApp()
        enabled_flag = nc.ocs("GET", "/ocs/v1.php/apps/app_api/ex-app/state")
        if enabled_flag:
            app_enabled.set()
            start_bg_task()
    except Exception as e:
        print(f"Failed to check the enabled state on startup, background task did not start: {e}", flush=True)
    yield


APP = FastAPI(lifespan=lifespan)


def background_thread_task(chains: dict):
    nc = NextcloudApp()

    provider_ids = set()
    task_type_ids = set()
    for chain_name, _ in chains.items():
        provider_ids.add("llm2:" + chain_name)
        (model, task) = chain_name.split(":", 1)
        task_type_ids.add(task)

    while True:
        if not app_enabled.is_set():
            sleep(5)
            break

        try:
            response = nc.providers.task_processing.next_task(list(provider_ids), list(task_type_ids))
            if not response:
                sleep(5)
                continue
        except (NextcloudException, httpx.RequestError) as e:
            print("Network error fetching the next task", e, flush=True)
            sleep(5)
            continue

        task = response["task"]
        provider = response["provider"]

        try:
            chain_name = provider["name"][5:]
            print(f"chain: {chain_name}", flush=True)
            chain_load = chains.get(chain_name)
            if chain_load is None:
                NextcloudApp().providers.task_processing.report_result(
                    task["id"], error_message="Requested model is not available"
                )
                continue

            chain = chain_load()
            print("Generating reply", flush=True)
            time_start = perf_counter()
            print(task.get("input").get("input"), flush=True)
            result = chain.invoke(task.get("input")).get("text")
            del chain
            print(f"reply generated: {round(float(perf_counter() - time_start), 2)}s", flush=True)
            print(result, flush=True)
            NextcloudApp().providers.task_processing.report_result(
                task["id"],
                {"output": str(result)},
            )
        except (NextcloudException, httpx.RequestError) as e:
            print("Network error:", e, flush=True)
            sleep(5)
        except Exception as e:  # noqa
            print("Error:", e, flush=True)
            try:
                nc = NextcloudApp()
                nc.log(LogLvl.ERROR, str(e))
                nc.providers.task_processing.report_result(task["id"], error_message=str(e))
            except (NextcloudException, httpx.RequestError) as net_err:
                print("Network error in reporting the error:", net_err, flush=True)

            sleep(5)


def start_bg_task():
    app_enabled.set()
    t = Thread(target=background_thread_task, args=(generate_chains(),))
    t.start()


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    global app_enabled
    print(f"enabled={enabled}", flush=True)

    chains = generate_chains()

    if enabled is True:
        for chain_name in chains:
            (model, task) = chain_name.split(":", 1)
            try:
                provider = TaskProcessingProvider(
                    id="llm2:" + chain_name,
                    name="Local Large language Model: " + model,
                    task_type=task,
                    expected_runtime=30,
                )
                await nc.providers.task_processing.register(provider)
                print(f"Registered {chain_name}", flush=True)
                start_bg_task()
            except Exception as e:
                print(f"Failed to register {model} - {task}, Error: {e}\n", flush=True)
                break
    else:
        for chain_name in chains:
            try:
                await nc.providers.task_processing.unregister("llm2:" + chain_name)
                print(f"Unregistered {chain_name}", flush=True)
            except Exception as e:
                print(f"Failed to unregister {chain_name}, Error: {e}\n", flush=True)
                break

        app_enabled.clear()

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
