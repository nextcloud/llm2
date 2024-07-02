"""Tha main module of the llm2 app
"""

import queue
import threading
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
TASK_LIST: queue.Queue = queue.Queue(maxsize=100)


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        while True:
            task = TASK_LIST.get(block=True)
            try:
                chain_name = task.get("chain")
                print(f"chain: {chain_name}", flush=True)
                chain_load = chains.get(chain_name)
                if chain_load is None:
                    NextcloudApp().providers.text_processing.report_result(
                        task["id"], error="Requested model is not available"
                    )
                    continue
                chain = chain_load()
                print("Generating reply", flush=True)
                time_start = perf_counter()
                print(task.get("prompt"))
                result = chain.invoke(task.get("prompt")).get("text")
                del chain
                print(f"reply generated: {round(float(perf_counter() - time_start), 2)}s", flush=True)
                print(result, flush=True)
                NextcloudApp().providers.text_processing.report_result(
                    task["id"],
                    str(result),
                )
            except Exception as e:  # noqa
                print(str(e), flush=True)
                nc = NextcloudApp()
                nc.log(LogLvl.ERROR, str(e))
                nc.providers.text_processing.report_result(task["id"], error=str(e))


class Input(pydantic.BaseModel):
    prompt: str
    task_id: int


@APP.post("/chain/{chain_name}")
async def tiny_llama(
    _nc: typing.Annotated[AsyncNextcloudApp, Depends(anc_app)],
    req: Input,
    chain_name=None,
):
    try:
        TASK_LIST.put({"prompt": req.prompt, "id": req.task_id, "chain": chain_name}, block=False)
    except queue.Full:
        return responses.JSONResponse(content={"error": "task queue is full"}, status_code=429)
    return responses.Response()


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    print(f"enabled={enabled}", flush=True)
    if enabled is True:
        for chain_name, _ in chains.items():
            (model, task) = chain_name.split(":", 2)
            try:
                await nc.providers.text_processing.register(
                    "llm2:"+chain_name, "Local Large language Model: " + model, "/chain/" + chain_name, task
                )
                print(f"Registering {model} - {task}", flush=True)
            except Exception as e:
                print(f"Failed to register", f"{model} - {task}", f"Error:", f"{e}\n", flush=True)
    else:
        for chain_name, chain in chains.items():
            (model, task) = chain_name.split(":", 2)
            await nc.providers.text_processing.unregister(model)
            print(f"Unregistering {model} - {task}", flush=True)
    return ""


if __name__ == "__main__":
    # print(os.environ["APP_HOST"], flush=True)
    # print(os.environ["APP_ID"], flush=True)
    # print(os.environ["APP_PORT"], flush=True)
    # print(os.environ["APP_SECRET"], flush=True)
    # print(os.environ["APP_VERSION"], flush=True)
    # print(os.environ["NEXTCLOUD_URL"], flush=True)
    # print(os.environ["APP_PERSISTENT_STORAGE"], flush=True)
    run_app("main:APP", log_level="trace")
