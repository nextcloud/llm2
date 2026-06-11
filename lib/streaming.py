from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable


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


@dataclass
class StreamContext:
    stream_result: Callable[[dict[str, Any]], None] | None = None
    progress_callback: Callable[[float], Any] | None = None
    stream_interval_seconds: float = 0.75
    current_output: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)
    _last_emit_at: float = field(default=0.0, init=False)

    @property
    def enabled(self) -> bool:
        return self.stream_result is not None

    def update_state(self, state: dict[str, Any] | None = None, *, force: bool = False, **extra: Any) -> None:
        if state:
            self.current_state.update(state)
        if extra:
            self.current_state.update(extra)
        self.emit(force=force)

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
        state: dict[str, Any] | None = None,
        force: bool = False,
        **extra_output: Any,
    ) -> None:
        self.current_output[key] = text
        if extra_output:
            self.current_output.update(extra_output)
        if state:
            self.current_state.update(state)
        self.emit(force=force)

    def emit(self, *, force: bool = False) -> None:
        if not self.stream_result:
            return

        if not self.current_output:
            return

        now = monotonic()
        if not force and self._last_emit_at and now - self._last_emit_at < self.stream_interval_seconds:
            return

        self.stream_result(dict(self.current_output))
        self._last_emit_at = now

    def set_progress(self, progress: float) -> Any:
        if self.progress_callback is None:
            return True
        return self.progress_callback(progress)


def run_runnable_with_streaming(
    runnable: Any,
    messages: list[Any],
    context: StreamContext | None = None,
    *,
    output_key: str = "output",
    state: dict[str, Any] | None = None,
    **extra_output: Any,
) -> str:
    if context and context.enabled:
        chunks: list[str] = []
        if state:
            context.update_state(state, force=True)

        for chunk in runnable.stream(messages):
            text_chunk = extract_text_content(chunk)
            if not text_chunk:
                continue
            chunks.append(text_chunk)
            context.update_text(
                "".join(chunks),
                key=output_key,
                state=state,
                **extra_output,
            )

        output = "".join(chunks)
        context.update_text(
            output,
            key=output_key,
            state=state,
            force=True,
            **extra_output,
        )
        return output

    return extract_text_content(runnable.invoke(messages))

