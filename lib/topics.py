"""
A chain that extracts topcis from a text
"""

from typing import Any, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from pydantic import Extra


class TopicsChain(Chain):
    """
    A topics chain
    """

    prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
        Find a maximum of 5 topics keywords for the following text:
        "
        {text}
        "
        List 5 topics of the above text as keywords separated by commas:
        """,
    )

    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
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
        out = self.llm.generate_prompt([self.prompt.format_prompt(text=inputs["text"])])
        text = out.generations[0][0].text

        return {self.output_key: text}

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
