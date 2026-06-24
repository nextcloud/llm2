# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
The main module of the llm2 app
"""
import asyncio
import os
import logging
import time
import traceback
from contextlib import asynccontextmanager
from json import JSONDecodeError
from threading import Event

from niquests import RequestException
from streaming import StreamContext
from task_processors import generate_task_processors, get_n_parallel, stop_all_servers
from fastapi import FastAPI
from nc_py_api import AsyncNextcloudApp, NextcloudApp, NextcloudException
from nc_py_api.ex_app import LogLvl, persistent_storage, run_app, set_handlers
from nc_py_api.ex_app.providers.task_processing import ShapeDescriptor, ShapeType, TaskProcessingProvider, \
    ShapeEnumValue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NextcloudTaskStreamResult:
    def __init__(self, nc: AsyncNextcloudApp, task_id: int, enabled: bool):
        self.nc = nc
        self.task_id = task_id
        self.enabled = enabled

    async def send(self, output: dict) -> None:
        if not self.enabled:
            return
        try:
            await self.nc.ocs(
                "POST",
                f"/ocs/v2.php/taskprocessing/tasks_provider/{self.task_id}/stream-result",
                json={"output": output},
            )
        except (NextcloudException, RequestException, JSONDecodeError) as e:
            logger.warning(f"Streaming intermediate task output failed for task {self.task_id}: {e}")
            self.enabled = False

    async def set_progress(self, progress: float) -> bool:
        try:
            await self.nc.providers.task_processing.set_progress(self.task_id, progress)
        except (NextcloudException, RequestException, JSONDecodeError) as e:
            logger.warning(f"Updating progress failed for task {self.task_id}: {e}")
            return False
        return True


async def log(nc: AsyncNextcloudApp, level, content):
    logger.log((level + 1) * 10, content)
    if level < LogLvl.WARNING:
        return
    try:
        await nc.log(level, content)
    except Exception:
        pass


models_to_fetch = {
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf": {"save_path": os.path.join(persistent_storage(), "Qwen3.5-9B-Q4_K_M.gguf")},
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/4f0c246f125fc7594238ebe7beb1435a8335f519/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": {"save_path": os.path.join(persistent_storage(), "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")},
    "https://huggingface.co/unsloth/Olmo-3-7B-Think-GGUF/resolve/25ad721cfffa2de24f43a51571dfd027108ca8ba/Olmo-3-7B-Think-Q4_K_M.gguf": {"save_path": os.path.join(persistent_storage(), "Olmo-3-7B-Think-Q4_K_M.gguf")},
}

# threading.Event: set from the async enabled_handler, checked in the async polling loop
app_enabled = Event()
# asyncio.Event: set from the async trigger_handler and on shutdown
trigger = asyncio.Event()
# Signals the polling loop to re-scan models on the next iteration — set by enabled_handler
# so newly available providers are picked up immediately instead of waiting up to SCAN_INTERVAL.
REFRESH_PROCESSORS = asyncio.Event()

NUM_RUNNING_TASKS = 0
NUM_RUNNING_TASKS_LOCK = asyncio.Lock()

# Per-model in-flight counter, bounded by each model's n_parallel. Keeps NC from
# marking tasks "running" while they are merely queued in llama-cpp-server.
MODEL_INFLIGHT: dict[str, int] = {}
MODEL_INFLIGHT_LOCK = asyncio.Lock()
MODEL_SLOT_FREED = asyncio.Event()

SHUTDOWN_EVENT = asyncio.Event()

try:
    CHECK_INTERVAL = float(os.getenv('TASK_POLLING_INTERVAL', '5'))
    if CHECK_INTERVAL <= 0:
        logger.warning("Invalid TASK_POLLING_INTERVAL env variable, falling back to default 5 seconds")
        CHECK_INTERVAL = 5
except (TypeError, ValueError):
    logger.warning("Invalid TASK_POLLING_INTERVAL env variable, falling back to default 5 seconds")
    CHECK_INTERVAL = 5

CHECK_INTERVAL_WITH_TRIGGER = 5 * 60
CHECK_INTERVAL_ON_ERROR = 10
SCAN_INTERVAL = 5 * 60


async def wait_for_tasks(interval: float | None = None) -> None:
    global CHECK_INTERVAL
    actual_interval = CHECK_INTERVAL if interval is None else interval
    try:
        await asyncio.wait_for(trigger.wait(), timeout=actual_interval)
        CHECK_INTERVAL = CHECK_INTERVAL_WITH_TRIGGER
    except asyncio.TimeoutError:
        pass
    trigger.clear()


def _model_of(processor_name: str) -> str:
    return processor_name.split(":", 1)[0]


async def available_provider_ids(task_processors: dict) -> list[str]:
    async with MODEL_INFLIGHT_LOCK:
        return [
            "llm2:" + name
            for name in task_processors
            if MODEL_INFLIGHT.get(_model_of(name), 0) < get_n_parallel(_model_of(name))
        ]


async def handle_task(task: dict, provider: dict, nc: AsyncNextcloudApp, task_processors: dict) -> None:
    global NUM_RUNNING_TASKS

    task_processor_name = provider["name"][5:]
    model_name = _model_of(task_processor_name)

    async with NUM_RUNNING_TASKS_LOCK:
        NUM_RUNNING_TASKS += 1

    try:
        await log(nc, LogLvl.INFO, f"Processing: {task_processor_name}")

        task_processor_loader = task_processors.get(task_processor_name)
        if task_processor_loader is None:
            await nc.providers.task_processing.report_result(
                task["id"], error_message="Requested model is not available"
            )
            return

        # task_processor_loader() calls generate_chat_model() which may start the llama-cpp-server
        # on first use (blocking I/O). Run it in the executor to avoid stalling the event loop.
        loop = asyncio.get_running_loop()
        processor = await loop.run_in_executor(None, task_processor_loader)

        stream_result = NextcloudTaskStreamResult(nc, task["id"], bool(task.get("preferStreaming")))
        stream_context = StreamContext(
            stream_result=stream_result.send if stream_result.enabled else None,
            progress_callback=stream_result.set_progress if stream_result.enabled else None,
        )

        time_start = time.perf_counter()
        result = await processor(task.get("input"), context=stream_context)
        await log(nc, LogLvl.INFO, f"Done in {round(time.perf_counter() - time_start, 2)}s: {result}")
        await nc.providers.task_processing.report_result(task["id"], result)

    except (NextcloudException, RequestException, JSONDecodeError) as e:
        tb_str = ''.join(traceback.format_exception(e))
        await log(nc, LogLvl.ERROR, f"Network error handling task: {tb_str}")
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(e))
        await log(nc, LogLvl.ERROR, f"Error handling task: {tb_str}")
        try:
            await nc.providers.task_processing.report_result(task["id"], error_message=str(e))
        except (NextcloudException, RequestException):
            pass
    finally:
        async with NUM_RUNNING_TASKS_LOCK:
            NUM_RUNNING_TASKS -= 1
        async with MODEL_INFLIGHT_LOCK:
            MODEL_INFLIGHT[model_name] = max(0, MODEL_INFLIGHT.get(model_name, 0) - 1)
        MODEL_SLOT_FREED.set()


async def background_task_loop() -> None:
    try:
        await _background_task_loop_inner()
    except Exception:
        # Without this, an unhandled exception in the bg task is silenced until shutdown awaits it,
        # leaving the polling loop dead with no log line — backend looks healthy but never claims tasks.
        logger.exception("background_task_loop crashed")
        raise


async def _background_task_loop_inner() -> None:
    nc = AsyncNextcloudApp()
    task_processors = generate_task_processors()
    last_scan = time.monotonic()

    task_type_ids = {n.split(":", 1)[1] for n in task_processors}

    while not app_enabled.is_set():
        await asyncio.sleep(5)

    async with asyncio.TaskGroup() as tg:
        while not SHUTDOWN_EVENT.is_set():
            if not app_enabled.is_set():
                await asyncio.sleep(5)
                continue

            if REFRESH_PROCESSORS.is_set() or time.monotonic() - last_scan >= SCAN_INTERVAL:
                task_processors = generate_task_processors(task_processors)
                task_type_ids = {n.split(":", 1)[1] for n in task_processors}
                last_scan = time.monotonic()
                REFRESH_PROCESSORS.clear()

            available = await available_provider_ids(task_processors)
            if not available:
                # Every model is at its n_parallel cap. Wait for a slot to free
                # rather than pulling tasks NC would otherwise mark "running".
                try:
                    await asyncio.wait_for(MODEL_SLOT_FREED.wait(), timeout=CHECK_INTERVAL)
                except asyncio.TimeoutError:
                    pass
                MODEL_SLOT_FREED.clear()
                continue

            try:
                response = await nc.providers.task_processing.next_task(
                    available, list(task_type_ids)
                )
            except (NextcloudException, RequestException, JSONDecodeError) as e:
                await log(nc, LogLvl.ERROR, f"Network error fetching the next task: {e}")
                await wait_for_tasks(CHECK_INTERVAL_ON_ERROR)
                continue

            if not response:
                async with NUM_RUNNING_TASKS_LOCK:
                    no_tasks_running = NUM_RUNNING_TASKS == 0
                if no_tasks_running:
                    await wait_for_tasks()
                else:
                    await asyncio.sleep(2)
                continue

            # Reserve the slot synchronously, before yielding to the loop again,
            # so the next available_provider_ids() call sees this task counted.
            pulled_model = _model_of(response["provider"]["name"][5:])
            async with MODEL_INFLIGHT_LOCK:
                MODEL_INFLIGHT[pulled_model] = MODEL_INFLIGHT.get(pulled_model, 0) + 1

            tg.create_task(handle_task(response["task"], response["provider"], nc, task_processors))
    # TaskGroup exits only after all spawned handle_task coroutines finish — graceful drain on shutdown


@asynccontextmanager
async def lifespan(_app: FastAPI):
    set_handlers(
        APP,
        enabled_handler,  # type: ignore
        models_to_fetch=models_to_fetch,
        trigger_handler=trigger_handler,
    )
    nc = NextcloudApp()
    if nc.enabled_state:
        app_enabled.set()
    bg = asyncio.get_event_loop().create_task(background_task_loop())
    yield
    print("\nSIGTERM received. Finishing in-flight tasks before shutdown...")
    SHUTDOWN_EVENT.set()
    trigger.set()  # wake up any wait_for_tasks() call so the loop checks SHUTDOWN_EVENT
    await bg
    stop_all_servers()


APP = FastAPI(lifespan=lifespan)


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    await log(nc, LogLvl.INFO, f"enabled={enabled}")

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
                    input_shape_enum_values={
                        "tone": [
                            ShapeEnumValue(name="Friendlier", value="friendlier"),
                            ShapeEnumValue(name="More formal", value="more formal"),
                            ShapeEnumValue(name="Funnier", value="funnier"),
                            ShapeEnumValue(name="More casual", value="more casual"),
                            ShapeEnumValue(name="More urgent", value="more urgent"),
                        ],
                    } if task == "core:text2text:changetone" else {},
                    optional_input_shape=[
                        ShapeDescriptor(name="memories", description="Memories to inject into the prompt", shape_type=ShapeType.LIST_OF_TEXTS)
                    ] if task == "core:text2text:chat" else [],
                    optional_output_shape=[
                        ShapeDescriptor(name="reasoning", description="Reasoning trace produced by the model, if any", shape_type=ShapeType.TEXT)
                    ] if task != "core:text2text:summary" else [],
                )
                await nc.providers.task_processing.register(provider)
                await log(nc, LogLvl.INFO, f"Registered {task_processor_name}")
                app_enabled.set()
                REFRESH_PROCESSORS.set()
            except Exception as e:
                await log(nc, LogLvl.ERROR, f"Failed to register {model} - {task}, Error: {e}\n")
                break
    else:
        app_enabled.clear()
        for task_processor_name in task_processors:
            try:
                await nc.providers.task_processing.unregister("llm2:" + task_processor_name)
                await log(nc, LogLvl.INFO, f"Unregistered {task_processor_name}")
            except Exception as e:
                await log(nc, LogLvl.ERROR, f"Failed to unregister {task_processor_name}, Error: {e}\n")
                break

    return ""


async def trigger_handler(providerId: str):
    trigger.set()


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
