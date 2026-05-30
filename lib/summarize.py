# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nc_py_api import NextcloudApp


class SummarizeProcessor:
    runnable: Runnable
    text_splitter: RecursiveCharacterTextSplitter

    system_prompt: str = "You're an AI assistant tasked with summarizing the text given to you by the user. Use a bullet list to summarize. Make sure to cover all topics of the text in your summary. Output only the summary without quotes, nothing else, especially no introductory or explanatory text."
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
Combine these summaries into one coherent summary. Preserve the most important information while making it concise and easy to understand.
The final summary should be in the same language as the original summaries. Detect the language of the original summaries. Use the same language for the final summary as the one you detected. Only output the final summary, no explanations or additional text. Do not mention the language used.

Summaries to combine:
{input}
"""
    )

    def __init__(self, runnable: Runnable, nc: NextcloudApp, task_id: int, n_ctx: int = 8000, max_tokens: int = 512):
        self.runnable = runnable
        self.nc = nc
        self.task_id = task_id
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens if max_tokens > 0 else 512
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n|\\.|\\?|\\!'],
            is_separator_regex=True,
            chunk_size=n_ctx/4,
            keep_separator=True,
            chunk_overlap=600,
            length_function=len,
        )

    def _invoke_progress(self, messages, max_tokens: int, idx: int, total_splits: int) -> str:
        # Stream the response and update progress

        start_pct = (idx / total_splits) * 100.0
        end_pct = ((idx + 1) / total_splits) * 100.0

        tokens_generated = 0
        full_response = ""
        total_range = end_pct - start_pct

        for chunk in self.runnable.stream(messages):
            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_response += token
            tokens_generated += 1

            fraction = min(1.0, tokens_generated / max_tokens)
            progress = start_pct + fraction * total_range
            self.nc.providers.task_processing.set_progress(self.task_id, progress)

        # Ensure the end percentage is set after completion
        self.nc.providers.task_processing.set_progress(self.task_id, end_pct)
        return full_response

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Split text if needed
        splits = self.text_splitter.split_text(inputs['input'])

        if len(splits) == 1:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.user_prompt.format(input=splits[0]))
            ]

            output = self._invoke_progress(messages, self.max_tokens, 0, 1)
            return {'output': output}

        # Process each split
        total_splits = len(splits)
        summaries = []

        for idx, split in enumerate(splits):
            

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.user_prompt.format(input=split))
            ]

            split_output = self._invoke_progress(messages, self.max_tokens, idx, total_splits)
            summaries.append(split_output)

        merge_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.merge_prompt.format(input="\n\n".join(summaries)))
        ]
        final_output = self.runnable.invoke(merge_messages)
        self.nc.providers.task_processing.set_progress(self.task_id, 100.0)
        return {'output': final_output.content}