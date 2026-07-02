# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain that improves a text based on instructions
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

from streaming import StreamContext, run_runnable_with_streaming


class ImproveProcessor:

    runnable: Runnable

    """
    An improve text processor
    """
    system_prompt: str = "You're an AI assistant tasked with improving the text given to you by the user based on specific instructions."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["input", "instructions"],
        template="""Improve the following text based on the provided instructions. Use the same language as the original text. Output only the improved text.

*INSTRUCTIONS*:
{instructions}

*TEXT*:

{input}

Do not mention the used language in your output. Here is your improved text in the same language:"""
    )

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    async def __call__(self, inputs: dict[str, Any], context: StreamContext | None = None) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt.format(
                input=inputs['input'],
                instructions=inputs['instructions']
            ))
        ]
        reasoning_sink: dict[str, str] = {}
        output = await run_runnable_with_streaming(
            self.runnable,
            messages,
            context,
            reasoning_sink=reasoning_sink,
        )
        return {'output': output, 'reasoning': reasoning_sink.get('reasoning', '')}
