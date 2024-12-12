# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A recursive summarize chain
"""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import Runnable


class SummarizeProcessor:
    """
    A summarization chain
    """

    system_prompt: str = "You're an AI assistant tasked with summarizing the text given to you by the user. If it's a long text, use bullet points to summarize. If the text is consisting of bullet points, also use bullet points to summarize."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
Summarize the following text. Detect the language of the text. Use the same language as the one you detected. Here is the text:

"
{text}
"

Output only the summary without quotes, nothing else, especially no introductory or explanatory text. Also, do not mention the language you used explicitly.
Here is your summary in the same language as the original text:
        """
    )


    runnable: Runnable
    n_ctx: int = 8000

    def __init__(self, runnable: Runnable, n_ctx: int = 8000):
        self.runnable = runnable
        self.n_ctx = n_ctx

    def __call__(
            self,
            inputs: dict[str, Any],
    ) -> dict[str, Any]:
        chunk_size = max(self.n_ctx * 0.7, 2048)

        text_splitter = CharacterTextSplitter(
            separator='\n\n|\\.|\\?|\\!', is_separator_regex=True, chunk_size=chunk_size*4, chunk_overlap=0, keep_separator=True)
        chunks = text_splitter.split_text(inputs['input'])
        print([len(chunk) for chunk in chunks])
        new_num_chunks = len(chunks)
        # first iteration outside of while loop
        old_num_chunks = new_num_chunks
        summaries = [self.runnable.invoke({"user_prompt": self.user_prompt.format_prompt(text=''.join(chunk)), "system_prompt": self.system_prompt}) for chunk in chunks]
        chunks = text_splitter.split_text('\n\n'.join(summaries))
        new_num_chunks = len(chunks)
        while (old_num_chunks > new_num_chunks):
            # now comes the while loop body
            old_num_chunks = new_num_chunks
            summaries = [self.runnable.invoke({"user_prompt": self.user_prompt.format_prompt(text=''.join(chunk)), "system_prompt": self.system_prompt}) for chunk in chunks]
            chunks = text_splitter.split_text('\n\n'.join(summaries))
            new_num_chunks = len(chunks)

        return {'output': '\n\n'.join(summaries)}
