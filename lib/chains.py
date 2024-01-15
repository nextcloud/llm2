from formalize import FormalizeChain
from simplify import SimplifyChain
from summarize import SummarizeChain
from headline import HeadlineChain
from topics import TopicsChain
from free_prompt import FreePromptChain
from langchain_community.llms import GPT4All
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

models = {
    #'llama2': GPT4All(model=dir_path + '/../models/llama-2-7b-chat.Q4_K_M.gguf'),
    'tiny_llama': GPT4All(model=dir_path + '/../models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf'),
   # 'leo_hessianai': GPT4All(model=dir_path + '/../models/leo-hessianai-13b-chat-bilingual.Q4_K_M.gguf')
}

chains = {}
for model_name, model in models.items():
    chains[model_name+':summarize'] = SummarizeChain(llm=model)
    chains[model_name+':simplify'] = SimplifyChain(llm=model)
    chains[model_name+':formalize'] = FormalizeChain(llm=model)
    chains[model_name+':headline'] = HeadlineChain(llm=model)
    chains[model_name+':topics'] = TopicsChain(llm=model)
    chains[model_name+':free_prompt'] = FreePromptChain(llm=model)