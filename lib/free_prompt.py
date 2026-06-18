# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from streaming import StreamContext, run_runnable_with_streaming


class FreePromptProcessor:
    """
	A free prompt chain with batch processing support
	"""

    runnable: Runnable
    system_prompt: str = "You're an AI assistant tasked with helping the user to the best of your ability."

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    async def __call__(
            self,
            inputs: dict[str, Any],
            context: StreamContext | None = None,
    ) -> dict[str, Any]:
        output = await run_runnable_with_streaming(self.runnable, [
            SystemMessage(self.system_prompt),
            HumanMessage(inputs['input'])
        ], context)
        return {'output': output}