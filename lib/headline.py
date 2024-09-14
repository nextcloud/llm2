# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chain to generate a headline for a text
"""

from typing import Any, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.chains import LLMChain


class HeadlineChain(Chain):
    """
    A headline chain
    """
    system_prompt = "You're an AI assistant tasked with finding a headline for the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
Give me the headline of the following text in its original language. Output only the headline.

"
{text}
"

Give me the headline of the above text in its original language. Do not output the language. Output only the headline without any quotes or additional punctuation.
        """
    )


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
        
        return self.llm_chain.invoke({"user_prompt": self.user_prompt.format_prompt(text=inputs['input']), "system_prompt": self.system_prompt})

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
