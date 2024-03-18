"""A chain that extracts topcis from a text
"""

from typing import Any, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate



class TopicsChain(Chain):
    """
    A topics chain
    """
    system_prompt = "You're an AI assistant tasked with finding the topic keywords of the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
        Find a maximum of 5 topics keywords for the following text:
        "
        {text}
        "
        List 5 topics of the above text as keywords separated by commas:
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
        return [self.output_key]

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
        
        return self.llm_chain.invoke({"user_prompt": self.user_prompt.format_prompt(text=inputs['text']), "system_prompt": self.system_prompt})

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
