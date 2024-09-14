# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A langchain chain to formalize text
"""

from typing import Any, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain

class ReformulateChain(Chain):
    """
    A reformulation chain
    """

    system_prompt = "You're an AI assistant tasked with reformulating the text given to you by the user."
    
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
Rewrite the following text and rephrase it:

"
{text}
"

Write the above text again and rephrase it using different words.
Also, Detect the language of the above text.
When rephrasing the text make sure to use the detected language in your rewritten text.
Output only the new text without quotes, nothing else, no introductory or explanatory text. Also do not explicitly mention the language you detected.
        """
    )
    # Multilingual output doesn't work with llama3.1


    """Prompt object to use."""
    llm_chain: LLMChain
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
            separator='\n\n|\\.|\\?|\\!', chunk_size=8000, chunk_overlap=0, keep_separator=True)
        texts = text_splitter.split_text(inputs['input'])
        outputs = self.llm_chain.apply([{"user_prompt": self.user_prompt.format_prompt(text=t), "system_prompt": self.system_prompt} for t in texts])
        texts = [output['text'] for output in outputs]

        return {self.output_key: '\n\n'.join(texts)}

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
