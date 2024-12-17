# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat chain
"""
import json
from typing import Any, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.runnables import Runnable


class ChatProcessor:
    """
    A free prompt chain
    """

    runnable: Runnable

    def __init__(self, runner: Runnable):
        self.runnable = runner

    def __call__(
            self,
            inputs: dict[str, Any],
    ) -> dict[str, str]:
        return {'output': self.runnable.invoke(
            [(message.role, message.content) for message in [json.loads(message) for message in inputs['history']]] + [('human', inputs['input'])]
        ).content}