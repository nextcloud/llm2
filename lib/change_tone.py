# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain that changes the tone of a text
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

class ChangeToneProcessor:

    runnable: Runnable

    """
    A change tone processor
    """
    system_prompt: str = "You're an AI assistant tasked with rewriting the text given to you by the user in another tone."
    user_prompt: BasePromptTemplate = PromptTemplate(
            input_variables=["text", "tone"],
            template="""Reformulate the following text in a " {tone} " tone in its original language without mentioning the language. Output only the reformulation, nothing else, no introductory sentence. Here is the text:

"
{text}
"

Output only the reformulated text, nothing else. Do not add an introductory sentence.
"""
    )

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, input_data: dict) -> dict[str, Any]:
        """Process a single input"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt.format_prompt(text=input_data['input'], tone=input_data['tone']).to_string())
        ]
        return {'output':self.runnable.invoke(messages).content }