# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain to proofread a text
"""

from typing import Any
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

from streaming import StreamContext, run_runnable_with_streaming


class ProofreadProcessor:
    """
    A proofreading chain
    """
    system_prompt: str = "You're an AI assistant tasked with proofreading the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text", "grammar_instructions"],
        template="""
{grammar_instructions}

"
{text}
"

Give me the list of all mistakes in the above text in its original language. Do not output the language. Output only the list in bullet points, nothing else, no introductory or explanatory text.
        """
    )

    runnable: Runnable

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    async def __call__(self, inputs: dict[str, Any], context: StreamContext | None = None) -> dict[str, Any]:
        grammar_instructions = "Detect all grammar and spelling mistakes of the following text in its original language. Output only the list of mistakes in bullet points."
        strictness = inputs.get("strictness", "standard")
        if strictness == "strict":
            grammar_instructions = "Detect every conceivable issue, including minor grammar rules, and list how to correct them in their original language. Also flag redundancy, and phrasing that could be clearer. Output only the list of mistakes in bullet points."
        elif strictness == "minimal":
            grammar_instructions = "Detect only grammatical and spelling errors that clearly affect meaning or readability in their original language and list how to correct them. Output only the list of mistakes in bullet points."
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt.format(
                text=inputs['input'],
                grammar_instructions=grammar_instructions
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
