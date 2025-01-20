# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain to proofread a text
"""

from typing import Any
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.runnables import Runnable


class ProofreadProcessor:
    """
    A proofreading chain
    """
    system_prompt: str = "You're an AI assistant tasked with proofreading the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
Detect all grammar and spelling mistakes of the following text in its original language. Output only the list of mistakes in bullet points.

"
{text}
"

Give me the list of all mistakes in the above text in its original language. Do not output the language. Output only the list in bullet points, nothing else, no introductory or explanatory text.
        """
    )

    runnable: Runnable

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(
            self,
            inputs: dict[str, Any],
    ) -> dict[str, Any]:
        output = self.runnable.invoke({"user_prompt": self.user_prompt.format_prompt(text=inputs['input']), "system_prompt": self.system_prompt})
        return {'output': output}