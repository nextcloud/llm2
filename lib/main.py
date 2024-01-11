"""Use the simple but clever model to test text recognition."""

import queue
import threading
import typing
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import Depends, FastAPI, responses
from huggingface_hub import snapshot_download
from transformers import pipeline

from nc_py_api import NextcloudApp, AsyncNextcloudApp
from nc_py_api.ex_app import anc_app, persistent_storage, run_app, set_handlers, LogLvl

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    set_handlers(APP, enabled_handler, models_to_fetch={MODEL_NAME: {"ignore_patterns": ["*.bin", "*onnx*"]}})
    t = BackgroundProcessTask()
    t.start()
    yield


APP = FastAPI(lifespan=lifespan)
TASK_LIST: queue.Queue = queue.Queue(maxsize=100)
PIPE: typing.Any = None


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        global PIPE

        while True:
            try:
                task = TASK_LIST.get(block=True, timeout=60 * 60)
                try:
                    if PIPE is None:
                        print("loading model")
                        time_start = perf_counter()
                        PIPE = pipeline(
                            "text-generation",
                            model=snapshot_download(
                                MODEL_NAME,
                                local_files_only=True,
                                cache_dir=persistent_storage(),
                            ),
                            device_map="auto",
                        )
                        print(f"model loaded: {perf_counter() - time_start}s")

                    messages = [
                        {
                            "role": "system",
                            "content": "story about",
                        },
                        {"role": "user", "content": task["prompt"]},
                    ]
                    print("tokenizing prompt")
                    time_start = perf_counter()
                    prompt = PIPE.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    print(f"prompt tokenized: {perf_counter() - time_start}s")

                    print("generating reply")
                    time_start = perf_counter()
                    r = PIPE(prompt, max_new_tokens=192, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)  # mypy
                    print(f"reply generated: {perf_counter() - time_start}s")
                    NextcloudApp().providers.text_processing.report_result(
                        task["id"], str(r[0]["generated_text"]).split(sep="<|assistant|>", maxsplit=1)[-1].strip()
                    )
                except Exception as e:  # noqa
                    print(str(e))
                    nc = NextcloudApp()
                    nc.log(LogLvl.ERROR, str(e))
                    nc.providers.text_processing.report_result(task["id"], error=str(e))
            except queue.Empty:
                if PIPE:
                    print("offloading model")
                PIPE = None


@APP.get("/tiny_llama")
async def tiny_llama(
    _nc: typing.Annotated[AsyncNextcloudApp, Depends(anc_app)],
    prompt: str,
    task_id: int,
):
    try:
        TASK_LIST.put({"prompt": prompt, "id": task_id}, block=False)
    except queue.Full:
        return responses.JSONResponse(content={"error": "task queue is full"}, status_code=429)
    return responses.Response()


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    print(f"enabled={enabled}")
    if enabled is True:
        await nc.providers.text_processing.register("TinyLlama", "TinyLlama", "/tiny_llama", "free_prompt")
    else:
        await nc.providers.text_processing.unregister("TinyLlama")
    return ""


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
