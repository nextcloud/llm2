"""Tha main module of the llm2 app
"""

import queue
import threading
import time
import typing
from contextlib import asynccontextmanager
from time import perf_counter

import pydantic
from chains import generate_chains
from fastapi import Depends, FastAPI, responses
from nc_py_api import AsyncNextcloudApp, NextcloudApp
from nc_py_api.ex_app import LogLvl, anc_app, run_app, set_handlers

chains = generate_chains()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    set_handlers(
        APP,
        enabled_handler,
    )
    t = BackgroundProcessTask()
    t.start()
    yield


APP = FastAPI(lifespan=lifespan)


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        nc = NextcloudApp()
        # Hack to workaround auth problems
        nc.set_user("admin")

        while True:
            response = nc.providers.task_processing.next_task("core:text2text")
            if not isinstance(response, dict):
                continue

            task = response["task"]

            try:
                chain_name = task["name"][4:]
                print(f"chain: {chain_name}", flush=True)
                chain_load = chains.get(chain_name)
                if chain_load is None:
                    NextcloudApp().providers.task_processing.report_result(
                        task["id"], error="Requested model is not available"
                    )
                    continue
                chain = chain_load()
                print("generating reply", flush=True)
                time_start = perf_counter()
                result = chain.invoke(task.get("prompt")).get("text")
                del chain
                print(f"reply generated: {perf_counter() - time_start}s", flush=True)
                print(result, flush=True)
                NextcloudApp().providers.task_processing.report_result(
                    task["id"],
                    str(result).split(sep="<|assistant|>", maxsplit=1)[-1].strip(),
                )
            except Exception as e:  # noqa
                print(str(e), flush=True)
                nc = NextcloudApp()
                nc.log(LogLvl.ERROR, str(e))
                nc.providers.task_processing.report_result(task["id"], error=str(e))


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    print(f"enabled={enabled}", flush=True)
    if enabled is True:
        for chain_name, _ in chains.items():
            (model, task) = chain_name.split(":", 2)
            await nc.providers.task_processing.register(
                "llm2:" + chain_name, "Local Large language Model: " + model, "core:text2text:" + task
            )
    else:
        for chain_name, chain in chains.items():
            await nc.providers.task_processing.unregister("llm2:" + chain_name)
    return ""


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
