# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Awaitable, Callable

# langchain-openai's converters drop `reasoning_content` (emitted by llama.cpp's
# deepseek reasoning format) from both streaming deltas and non-streaming
# responses. Patch them to preserve it under additional_kwargs so downstream
# code can read it back out.
try:
    from langchain_openai.chat_models import base as _lc_openai_base

    _original_delta_to_chunk = _lc_openai_base._convert_delta_to_message_chunk
    _original_dict_to_message = _lc_openai_base._convert_dict_to_message

    def _delta_to_chunk_with_reasoning(_dict, default_class):
        chunk = _original_delta_to_chunk(_dict, default_class)
        reasoning = _dict.get("reasoning_content") if hasattr(_dict, "get") else None
        if reasoning and hasattr(chunk, "additional_kwargs"):
            chunk.additional_kwargs["reasoning_content"] = reasoning
        return chunk

    def _dict_to_message_with_reasoning(_dict):
        message = _original_dict_to_message(_dict)
        reasoning = _dict.get("reasoning_content") if hasattr(_dict, "get") else None
        if reasoning and hasattr(message, "additional_kwargs"):
            message.additional_kwargs["reasoning_content"] = reasoning
        return message

    _lc_openai_base._convert_delta_to_message_chunk = _delta_to_chunk_with_reasoning
    _lc_openai_base._convert_dict_to_message = _dict_to_message_with_reasoning
except Exception:
    pass


def extract_text_content(value: Any) -> str:
    if value is None:
        return ""

    content = getattr(value, "content", value)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts)

    return str(content)


def extract_reasoning_content(value: Any) -> str:
    if value is None:
        return ""

    additional_kwargs = getattr(value, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        reasoning = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning, str):
            return reasoning

    if isinstance(value, dict):
        reasoning = value.get("reasoning_content")
        if isinstance(reasoning, str):
            return reasoning

    return ""


@dataclass
class StreamContext:
    stream_result: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None
    progress_callback: Callable[[float], Awaitable[Any] | Any] | None = None
    stream_interval_seconds: float = 0.75
    current_output: dict[str, Any] = field(default_factory=dict)
    _last_emit_at: float = field(default=0.0, init=False)
    _last_emitted_output: dict[str, Any] | None = field(default=None, init=False)

    @property
    def enabled(self) -> bool:
        return self.stream_result is not None

    def update_output(self, output: dict[str, Any] | None = None, *, force: bool = False, **extra: Any) -> None:
        if output:
            self.current_output.update(output)
        if extra:
            self.current_output.update(extra)
        self.emit(force=force)

    def update_text(
        self,
        text: str,
        *,
        key: str = "output",
        force: bool = False,
        **extra_output: Any,
    ) -> None:
        self.current_output[key] = text
        if extra_output:
            self.current_output.update(extra_output)
        self.emit(force=force)

    def emit(self, *, force: bool = False) -> None:
        if not self.stream_result:
            return

        if not self.current_output:
            return

        now = monotonic()
        if not force and self._last_emit_at and now - self._last_emit_at < self.stream_interval_seconds:
            return

        payload = dict(self.current_output)
        if payload == self._last_emitted_output:
            return

        if asyncio.iscoroutinefunction(self.stream_result):
            asyncio.ensure_future(self.stream_result(payload))
        else:
            self.stream_result(payload)
        self._last_emit_at = now
        self._last_emitted_output = payload

    def set_progress(self, progress: float) -> Any:
        if self.progress_callback is None:
            return True
        result = self.progress_callback(progress)
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)
            return True
        return result


async def run_runnable_with_streaming(
    runnable: Any,
    messages: list[Any],
    context: StreamContext | None = None,
    *,
    output_key: str = "output",
    stream_text_transform: Callable[[str], str] | None = None,
    stream_payload_transform: Callable[[str], dict[str, Any] | None] | None = None,
    suppress_empty_stream_updates: bool = False,
    reasoning_sink: dict[str, str] | None = None,
    **extra_output: Any,
) -> str:
    capture_reasoning = reasoning_sink is not None

    if context and context.enabled:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        async for chunk in runnable.astream(messages):
            text_chunk = extract_text_content(chunk)
            reasoning_chunk = extract_reasoning_content(chunk) if capture_reasoning else ""

            if not text_chunk and not reasoning_chunk:
                continue

            if text_chunk:
                text_parts.append(text_chunk)
            if reasoning_chunk:
                reasoning_parts.append(reasoning_chunk)

            output = "".join(text_parts)
            reasoning = "".join(reasoning_parts)

            stream_extra = dict(extra_output)
            if capture_reasoning and reasoning:
                stream_extra["reasoning"] = reasoning

            if stream_payload_transform:
                streamed_payload = stream_payload_transform(output)
                if streamed_payload is None:
                    if capture_reasoning and reasoning:
                        context.update_output({"reasoning": reasoning, **extra_output})
                    continue
                if stream_extra:
                    streamed_payload.update(stream_extra)
                if suppress_empty_stream_updates and not streamed_payload:
                    continue
                context.update_output(streamed_payload)
                continue

            streamed_output = stream_text_transform(output) if stream_text_transform else output
            if suppress_empty_stream_updates and streamed_output == "":
                if capture_reasoning and reasoning:
                    context.update_output({"reasoning": reasoning, **extra_output})
                continue
            context.update_text(
                streamed_output,
                key=output_key,
                **stream_extra,
            )

        output = "".join(text_parts)
        reasoning_final = "".join(reasoning_parts)
        if capture_reasoning:
            reasoning_sink["reasoning"] = reasoning_final

        stream_extra_final = dict(extra_output)
        if capture_reasoning and reasoning_final:
            stream_extra_final["reasoning"] = reasoning_final

        if stream_payload_transform:
            streamed_payload = stream_payload_transform(output)
            if streamed_payload is None:
                if capture_reasoning and reasoning_final:
                    context.update_output({"reasoning": reasoning_final, **extra_output}, force=True)
                return output
            if stream_extra_final:
                streamed_payload.update(stream_extra_final)
            if not suppress_empty_stream_updates or streamed_payload:
                context.update_output(streamed_payload, force=True)
            return output

        streamed_output = stream_text_transform(output) if stream_text_transform else output
        context.update_text(
            streamed_output,
            key=output_key,
            force=True,
            **stream_extra_final,
        )
        return output

    result = await runnable.ainvoke(messages)
    if capture_reasoning:
        reasoning_sink["reasoning"] = extract_reasoning_content(result)
    return extract_text_content(result)
