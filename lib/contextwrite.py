# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain that changes the tone of a text
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

class ContextWriteProcessor:

    runnable: Runnable

    """
    A context write processor
    """
    system_prompt: str = "You're an AI assistant tasked with reformulating the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["style_input", "source_input"],
        template="""
You're a professional copywriter tasked with copying an instructed or demonstrated *WRITING STYLE* and writing a text on the provided *SOURCE MATERIAL*.
*WRITING STYLE*:
{style_input}

*SOURCE MATERIAL*:
{source_input}

Now write a text in the same style detailed or demonstrated under *WRITING STYLE* using the *SOURCE MATERIAL* as source of facts and instruction on what to write about.
Do not invent any facts or events yourself. Also, use the *WRITING STYLE* as a guide for how to write the text ONLY and not as a source of facts or events.
Detect the language used in the *SOURCE_MATERIAL*. Make sure to use the detected language in your text. Do not mention the language you detected explicitly.
Only output the newly written text without quotes, nothing else, no introductory or explanatory text.
        """
    )
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt.format(
                style_input=inputs['style_input'],
                source_input=inputs['source_input']
            ))
        ]
        output = self.runnable.invoke(messages)
        return {'output': output.content}