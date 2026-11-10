# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helpers for fetching Task Processing input files (images, etc.)."""
from __future__ import annotations

import base64
from typing import Any

from nc_py_api import AsyncNextcloudApp

VALID_TEXT_MIME_TYPES = frozenset({
    "application/javascript",
    "application/typescript",
    "message/rfc822",
    "application/x-sql",
    "application/x-scala",
    "application/x-rust",
    "application/x-powershell",
    "application/x-patch",
    "application/x-php",
    "application/x-httpd-php",
    "application/x-httpd-php-source",
    "application/json",
    "application/x-bash",
    "application/x-protobuf",
    "application/x-terraform",
    "application/x-toml",
    "application/graphql",
    "application/x-graphql",
    "application/x-ndjson",
    "application/json5",
    "application/x-json5",
    "application/toml",
    "application/x-yaml",
    "application/yaml",
    "application/x-awk",
    "application/x-subrip",
    "application/csv",
})

SUPPORTED_INPUT_AUDIO_FORMATS = {
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}



def bytes_to_data_url(data: bytes, mime: str) -> str:
    content_type = mime.split(";")[0].strip() or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def is_text_mime(mime: str) -> bool:
    return mime.startswith("text/") or mime in VALID_TEXT_MIME_TYPES


async def fetch_file_bytes(nc: AsyncNextcloudApp, file_id: int | str) -> dict[str, Any]:
    """Download a Nextcloud file by ID using the public Files API.

    MIME type comes from FsNode metadata (same source Nextcloud uses for Content-Type).
    """
    node = await nc.files.by_id(int(file_id))
    if node is None:
        raise RuntimeError(f"File not found: {file_id}")
    data = await nc.files.download(node)
    if not data:
        raise RuntimeError(f"Empty file content for file {file_id}")
    mime = (node.info.mimetype or "").split(";")[0].strip()
    if not mime:
        raise RuntimeError(f"Missing mimetype metadata for file {file_id}")
    return {
        "data": data,
        "mime": mime,
        "data_url": bytes_to_data_url(data, mime),
        "name": node.name,
        "file_id": int(file_id),
    }


def file_reference_text(name: str, file_id: int) -> str:
    return f'Attached file: "{name}" (file ID: {file_id})'


async def build_attachment_content_parts(
    nc: AsyncNextcloudApp,
    file_id: int | str,
    modalities: list[str],
) -> list[dict[str, Any]]:
    """Fetch a file and turn it into OpenAI-style message content parts.

    Every file is labelled with its name and file ID so the model can pass the ID to tools.
    Files whose content cannot be given to the model directly only get the label.
    """
    fetched = await fetch_file_bytes(nc, file_id)
    mime = fetched["mime"]
    reference = file_reference_text(fetched["name"], fetched["file_id"])
    if mime.startswith("image/") and "vision" in modalities:
        return [
            {"type": "text", "text": reference},
            {"type": "image_url", "image_url": {"url": fetched["data_url"]}},
        ]
    elif mime.startswith("audio/") and "audio" in modalities and mime in SUPPORTED_INPUT_AUDIO_FORMATS:
        return [
            {"type": "text", "text": reference},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(fetched["data"]).decode("ascii"),
                    "format": SUPPORTED_INPUT_AUDIO_FORMATS[mime],
                },
            },
        ]
    elif is_text_mime(mime):
        try:
            text_body = fetched["data"].decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return [{
                "type": "text",
                "text": f"Filename:{fetched['name']}\nFile ID:{fetched['file_id']}\nContent:\n{text_body}",
            }]
    return [{
        "type": "text",
        "text": f"{reference}\n(content of type {mime} not available directly; use tools with the file ID)",
    }]
