# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter

from streaming import StreamContext, run_runnable_with_streaming


class SummarizeProcessor:
    runnable: Runnable
    text_splitter: RecursiveCharacterTextSplitter

    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["input"],
        template="""
Summarize the following text. Detect the language of the text. Use the same language as the one you detected. Also, do not mention the language you used explicitly. Here is the text:

Text to summarize:

{input}"""
    )

    merge_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["input"],
        template="""
Combine these summaries into one coherent summary. Preserve the most important information while making it concise.
The final summary should be in the same language as the original summaries. Detect the language of the original summaries. Use the same language for the final summary as the one you detected. Only output the final summary, no explanations or additional text. Do not mention the language used.

Summaries to combine:
{input}
"""
    )

    def __init__(self, runnable: Runnable, n_ctx: int = 8000):
        self.runnable = runnable
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n|\\.|\\?|\\!'],
            is_separator_regex=True,
            chunk_size=n_ctx/4,
            keep_separator=True,
            chunk_overlap=600,
            length_function=len,
        )

    def _build_system_prompt(self, format: str, complexity: str) -> str:
        prompt = (
            "You're an AI assistant tasked with summarizing the text given to you by the user. "
            "Make sure to cover all topics of the text in your summary. "
            "Output only the summary without quotes, nothing else, especially no introductory or explanatory text. "
        )
        if format == "paragraph":
            prompt += "Return the summary as a paragraph. "
        elif format == "bullet_points":
            prompt += "Return the summary as a list of bullet points. "
        elif format == "sentence":
            prompt += "Return the summary as a single sentence. Do not include more than one sentence. "
        if complexity == "complex":
            prompt += "Use complex language and vocabulary appropriate for an expert in the subject. "
        elif complexity == "simple":
            prompt += "Use simple language and vocabulary appropriate for a 5 year old. "
        return prompt

    async def __call__(self, inputs: dict[str, Any], context: StreamContext | None = None) -> dict[str, Any]:
        system_prompt = self._build_system_prompt(
            inputs.get("format", "auto"),
            inputs.get("complexity", "medium"),
        )

        # Split text if needed
        splits = self.text_splitter.split_text(inputs['input'])
        total_splits = len(splits)

        if len(splits) == 1:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=self.user_prompt.format(input=splits[0]))
            ]
            output = await run_runnable_with_streaming(
                self.runnable,
                messages,
                context,
            )
            return {'output': output}

        # Process each split
        summaries = []
        for index, split in enumerate(splits, start=1):
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=self.user_prompt.format(input=split))
            ]
            output = await self.runnable.ainvoke(messages)
            summaries.append(output.content)
            if context:
                context.set_progress(index / (total_splits + 1) * 50)

        # Merge summaries
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self.merge_prompt.format(input="\n\n".join(summaries)))
        ]
        if context:
            context.set_progress(total_splits / (total_splits + 1) * 50 + 50)

        final_output = await run_runnable_with_streaming(
            self.runnable,
            messages,
            context,
        )

        if context:
            context.set_progress(100)

        return {'output': final_output}
