# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable


class FreePromptProcessor:
    """
	A free prompt chain with batch processing support
	"""

    runnable: Runnable
    system_prompt: str = "You're an AI assistant tasked with helping the user to the best of your ability."

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(
            self,
            inputs: dict[str, Any],
    ) -> dict[str, Any]:
        output = self.runnable.invoke([
            SystemMessage(self.system_prompt),
            HumanMessage(inputs['input'])
        ]).content
        return {'output': output}