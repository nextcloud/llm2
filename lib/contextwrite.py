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

class ContextWriteChain(Chain):
    """
    A reformulation chain
    """

    system_prompt = "You're an AI assistant tasked with reformulating the text given to you by the user."
    
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["style_input", "source_input"],
        template="""
You're a professional copywriter tasked with copying an instructed or demonstrated *WRITING STYLE* and writing a text on the provided *SOURCE MATERIAL*.
*WRITING STYLE*:
{style_input}

*SOURCE MATERIAL*:
{source_input}

Now write a text in the same style detailed or demonstrated under *WRITING STYLE* using the *SOURCE MATERIAL* as source of facts and instruction on what to write about.
Do not invent any facts or events yourself. Also, use the *WRITING STYLE* as a guide for how to write the text ONLY and not as a source of facts or events.
Detect the language used in the *SOURCE_MATERIAL*. Make sure to use the detected language in your text. Do not mention the language you detected explicitly.
Only output the newly written text without quotes, nothing else, no introductory or explanatory text.
        """
    )
    # Multilingual output doesn't work with llama3.1
    # Task doesn't work with llama 3.1


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
        return ['style_input', 'source_input']

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

        return self.llm_chain.invoke({"user_prompt": self.user_prompt.format_prompt(style_input=inputs['style_input'], source_input=inputs['source_input']), "system_prompt": self.system_prompt})

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
