# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain to generate a headline for a text
"""

from typing import Any
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.runnables import Runnable


class HeadlineProcessor:
    """
    A headline chain
    """
    system_prompt: str = "You're an AI assistant tasked with finding a headline for the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
Give me the headline of the following text in its original language. Output only the headline.

"
{text}
"

Give me the headline of the above text in its original language. Do not output the language. Output only the headline without any quotes or additional punctuation.
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