# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A free prompt chain
"""

from typing import Any

from langchain_core.runnables import Runnable


class FreePromptProcessor:
    """
    A free prompt chain
    """

    runnable: Runnable

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(
            self,
            inputs: dict[str, Any],
    ) -> dict[str, Any]:
        output = self.runnable.invoke({"user_prompt": inputs['input'], "system_prompt": "You're an AI assistant tasked with helping the user to the best of your ability."})
        return {'output': output}
