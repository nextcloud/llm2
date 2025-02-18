# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable


class SimplifyProcessor:
    """
    A simplify chain
    """

    system_prompt = "You're an AI assistant tasked with simplifying the text given to you by the user."

    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
Rewrite and rephrase the following text in the same language to make it easier to understand, so that a 5-year-old child can understand it.

"
{text}
"

Rewrite and rephrase the above text to make it easier to understand, so that a 5-year-old child can understand it.
Describe difficult concepts in the text instead of using jargon terms directly. Do not make up anything new that is not in the original text.
Also, detect the language of the text. Make sure to use the same language as the text in your simplification.
Output only the new, rewritten text without quotes, nothing else. Do not mention the language of the text explicitly. Do not add any introductory or explanatory text.
        """
    )

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
