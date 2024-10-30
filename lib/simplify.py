# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A simplify chain
"""

from typing import Any, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain


class SimplifyChain(Chain):
    """A summarization chain
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
    # Multilingual output doesn't work with llama3.1

    llm_chain: LLMChain
    chunk_size: int = 8000
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = 'forbid'
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ['input']

    @property
    def output_keys(self) -> list[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        if not {"user_prompt", "system_prompt"} == set(self.llm_chain.input_keys):
            raise ValueError("llm_chain must have input_keys ['user_prompt', 'system_prompt']")
        if not self.llm_chain.output_keys == [self.output_key]:
            raise ValueError(f"llm_chain must have output_keys [{self.output_key}]")
        
        text_splitter = CharacterTextSplitter(
            separator="\n\n|\\.|\\?|\\!", chunk_size=self.chunk_size, chunk_overlap=0, keep_separator=True
        )
        texts = text_splitter.split_text(inputs["input"])
        outputs = self.llm_chain.apply([{"user_prompt": self.user_prompt.format_prompt(text=t), "system_prompt": self.system_prompt} for t in texts])
        texts = [output['text'] for output in outputs]

        return {self.output_key: "\n\n".join(texts)}

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
