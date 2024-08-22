"""Tha main module of the llm2 app
"""

import threading
from contextlib import asynccontextmanager
from time import perf_counter, sleep

import httpx
from fastapi import FastAPI
from nc_py_api import AsyncNextcloudApp, NextcloudApp, NextcloudException
from nc_py_api.ex_app import LogLvl, run_app, set_handlers
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider

from chains import generate_chains

chains = generate_chains()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    set_handlers(
        APP,
        enabled_handler, # type: ignore
    )
    t = BackgroundProcessTask()
    t.start()
    yield


APP = FastAPI(lifespan=lifespan)


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        nc = NextcloudApp()

        provider_ids = set()
        task_type_ids = set()
        for chain_name, _ in chains.items():
            provider_ids.add("llm2:" + chain_name)
            (model, task) = chain_name.split(":", 1)
            task_type_ids.add(task)

        while True:
            try:
                enabled_flag = nc.ocs("GET", "/ocs/v1.php/apps/app_api/ex-app/state")
                if not enabled_flag:
                    sleep(10)
                    continue

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


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    print(f"enabled={enabled}", flush=True)
    if enabled is True:
        for chain_name, _ in chains.items():
            (model, task) = chain_name.split(":", 1)
            try:
                provider = TaskProcessingProvider(
                    id="llm2:" + chain_name,
                    name="Local Large language Model: " + model,
                    task_type=task,
                    expected_runtime=30,
                )
                await nc.providers.task_processing.register(provider)
                print(f"Registering {chain_name}", flush=True)
            except Exception as e:
                print(f"Failed to register", f"{model} - {task}", f"Error:", f"{e}\n", flush=True)
    else:
        for chain_name, chain in chains.items():
            await nc.providers.task_processing.unregister("llm2:" + chain_name)
            print(f"Unregistering {chain_name}", flush=True)
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
