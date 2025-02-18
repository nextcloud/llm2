# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain that extracts topcis from a text
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

class TopicsProcessor():

    runnable: Runnable

    """
    A topics chain
    """
    system_prompt: str = "You're an AI assistant tasked with finding the topic keywords of the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
            input_variables=["text"],
            template="""
    Find a maximum of 5 topics keywords for the following text:
    
    "
    {text}
    "
    
    List 5 topics of the above text as keywords separated by commas. Output only the topics, nothing else, no introductory sentence. Use the same language for the topics as the text.
    """
    )

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt.format_prompt(text=inputs['input']).to_string())
        ]
        output = self.runnable.invoke(messages).content
        return {'output': output}