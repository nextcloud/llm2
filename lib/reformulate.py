# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable


class ReformulateProcessor:
    """
    A reformulate chain
    """
    system_prompt = "You're an AI assistant tasked with reformulating the text given to you by the user."

    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""Rewrite the following text and rephrase it:

"
{text}
"

Write the above text again and rephrase it using different words.
Also, Detect the language of the above text.
When rephrasing the text make sure to use the detected language in your rewritten text.
Output only the new text without quotes, nothing else, no introductory or explanatory text. Also do not explicitly mention the language you detected.
""")

    runnable: Runnable

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt.format(
                text=inputs['input']
            ))
        ]
        output = self.runnable.invoke(messages)
        return {'output': output.content}