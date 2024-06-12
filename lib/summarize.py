"""A recursive summarize chain
"""

from typing import Any, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain


class SummarizeChain(Chain):
    """
    A summarization chain
    """

    system_prompt = "You're an AI assistant tasked with summarizing the text given to you by the user."
    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
        Summarize the following text. Detect the language of the text. Use the same language as the text. Here is the text:
        "
        {text}
        "
        Output only the summary. Here is your summary in the same language as the text:
        """
    )


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
        
        text_splitter = CharacterTextSplitter(
            separator='\n\n|\\.|\\?|\\!', chunk_size=8000, chunk_overlap=0, keep_separator=True)
        texts = text_splitter.split_text(inputs['text'])
        while sum([len(text) for text in texts]) > max(len(inputs['text']) * 0.2, 1000):  # 2000 chars summary per 10.000 chars original text
            docs = [texts[i:i + 3] for i in range(0, len(texts), 3)]
            outputs = self.llm_chain.apply([{"user_prompt": self.user_prompt.format_prompt(text=''.join(doc)), "system_prompt": self.system_prompt} for doc in docs])
            texts = [output['text'] for output in outputs]

        return {self.output_key: '\n\n'.join(texts)}

    @property
    def _chain_type(self) -> str:
        return "summarize_chain"
