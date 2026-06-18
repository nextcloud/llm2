# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat chain
"""
import json
from typing import Any

from langchain_core.runnables import Runnable

from streaming import StreamContext, run_runnable_with_streaming


class ChatProcessor:
    """
    A free prompt chain
    """

    runnable: Runnable

    def __init__(self, runner: Runnable):
        self.runnable = runner

    async def __call__(
            self,
            inputs: dict[str, Any],
            context: StreamContext | None = None,
    ) -> dict[str, str]:
        system_prompt = inputs['system_prompt']
        if inputs.get('memories'):
            system_prompt += "\n\nYou can remember things from other conversations with the user. If they are relevant, take into account the following memories: \n" + "\n\n".join(inputs['memories']) + "\n\n"
        messages = [('human', system_prompt)] + [
            (message['role'], message['content'])
            for message in [json.loads(message) for message in inputs['history']]
        ] + [('human', inputs['input'])]
        return {
            'output': await run_runnable_with_streaming(
                self.runnable,
                messages,
                context,
            )
        }
