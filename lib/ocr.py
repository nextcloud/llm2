# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Processor for core:image2text:ocr — vision-based OCR."""
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from task_files import fetch_file_bytes
from streaming import StreamContext, run_runnable_with_streaming
from multimodal_chatwithtools import MAX_ATTACHMENTS_COUNT


class OcrProcessor:
    """Extract text from one or more images using a vision-capable chat model."""

    runnable: Runnable
    system_prompt: str = (
        "You're an OCR assistant. "
        "Extract all text visible in the provided image. "
        "Output only the extracted text, nothing else."
    )
    user_prompt: str = "Extract all text from this image. Reply with only the extracted text."

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    async def __call__(
            self,
            inputs: dict[str, Any],
            context: StreamContext | None = None,
    ) -> dict[str, Any]:
        if context is None or context.nc is None:
            raise ValueError("StreamContext with Nextcloud client is required for OCR")

        files = inputs.get("input") or []
        if not files:
            raise ValueError("core:image2text:ocr requires at least one file")
        if len(files) > MAX_ATTACHMENTS_COUNT:
            raise ValueError(f"Too many files (max {MAX_ATTACHMENTS_COUNT})")

        texts: list[str] = []
        for file_id in files:
            fetched = await fetch_file_bytes(context.nc, file_id)
            if not fetched["mime"].startswith("image/"):
                raise ValueError(f"File MIME type {fetched['mime']} is not supported for OCR")

            output = await run_runnable_with_streaming(
                self.runnable,
                [
                    SystemMessage(self.system_prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": self.user_prompt},
                        {"type": "image_url", "image_url": {"url": fetched["data_url"]}},
                    ]),
                ],
                context,
                stream_payload_transform=lambda partial: {"output": [*texts, partial]},
            )
            texts.append(output)

        return {
            "output": texts,
        }
