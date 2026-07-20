# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helpers for fetching Task Processing input files (images, etc.)."""
from __future__ import annotations

import base64
import logging
from typing import Any

from nc_py_api import AsyncNextcloudApp

logger = logging.getLogger(__name__)


def bytes_to_data_url(data: bytes, mime: str) -> str:
    content_type = mime.split(";")[0].strip() or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


async def fetch_file_bytes(nc: AsyncNextcloudApp, file_id: int | str) -> tuple[bytes, str]:
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
    data_url = bytes_to_data_url(data, mime)
    return {"data": data, "mime": mime, "data_url": data_url}