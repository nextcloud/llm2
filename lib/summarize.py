# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

    def __init__(self, runnable: Runnable, n_ctx: int = 8000):
        self.runnable = runnable
        self.text_splitter = RecursiveCharacterTextSplitter(
            separator='\n\n|\\.|\\?|\\!',
            is_separator_regex=True,
            chunk_size=n_ctx*4,
            keep_separator=True,
            chunk_overlap=600,
            length_function=len,
        )

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Split text if needed
        splits = self.text_splitter.split_text(inputs['input'])

        if len(splits) == 1:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.user_prompt.format(input=splits[0]))
            ]
            output = self.runnable.invoke(messages)
            return {'output': output.content}

        # Process each split
        summaries = []
        for split in splits:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.user_prompt.format(input=split))
            ]
            output = self.runnable.invoke(messages)
            summaries.append(output.content)

        # Merge summaries
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.merge_prompt.format(input="\n\n".join(summaries)))
        ]
        final_output = self.runnable.invoke(messages)
        return {'output': final_output.content}