# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Processor for core:analyze-images — multimodal vision Q&A."""
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from task_files import fetch_file_bytes
from streaming import StreamContext, run_runnable_with_streaming


class AnalyzeImagesProcessor:
    """Ask a question about one or more images using a vision-capable chat model."""

    runnable: Runnable
    system_prompt: str = (
        "You're an AI assistant that analyzes images. "
        "Answer the user's question about the provided image(s) helpfully and accurately."
    )

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    async def __call__(
            self,
            inputs: dict[str, Any],
            context: StreamContext | None = None,
    ) -> dict[str, Any]:
        if context is None or context.nc is None:
            raise ValueError("StreamContext with Nextcloud client is required for analyze-images")

        question = inputs.get("input") or ""
        images = inputs.get("images") or []
        if not images:
            raise ValueError("core:analyze-images requires at least one image")
        if len(images) > 10:
            raise ValueError("Too many images")
        images = [await fetch_file_bytes(context.nc, image) for image in images]

        content: list[dict[str, Any]] = [{"type": "text", "text": question}]
        for image in images:
            if not image["mime"].startswith("image/"):
                raise ValueError(f"Image MIME type {image['mime']} is not supported")
            content.append({"type": "image_url", "image_url": {"url": image["data_url"]}})

        reasoning_sink: dict[str, str] = {}
        output = await run_runnable_with_streaming(
            self.runnable,
            [
                SystemMessage(self.system_prompt),
                HumanMessage(content=content),
            ],
            context,
            reasoning_sink=reasoning_sink,
        )
        return {
            "output": output,
            "reasoning": reasoning_sink.get("reasoning", ""),
        }
