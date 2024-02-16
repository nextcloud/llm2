"""Registers all chains based on the models/ directory contents
"""

import os

from formalize import FormalizeChain
from free_prompt import FreePromptChain
from headline import HeadlineChain
from langchain_community.llms import GPT4All
from langchain.chains import LLMChain
from simplify import SimplifyChain
from summarize import SummarizeChain
from topics import TopicsChain
from langchain.prompts import PromptTemplate
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
models_folder_path = os.path.join(dir_path , "../models/")

def get_model_config(file_name):
    file_name = file_name.split('.gguf')[0]
    if os.path.exists(os.path.join(models_folder_path, file_name + ".json")):
        model_config_path = os.path.join(models_folder_path, file_name + ".json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config_path = os.path.join(dir_path, "default_config", "config.json")
        
        with open(model_config_path, "r") as f:
            default_config = json.load(f)
        
        if file_name in default_config:
            model_config = default_config[file_name]
        else:
            model_config = default_config['default']
    return model_config


def generate_llm_chain(file_name):
    model_config = get_model_config(file_name)
    
    path = os.path.join(models_folder_path, file_name)
    
    try:
        llm = GPT4All(model=path, device="gpu", **model_config['gpt4all_config'])
    except:
        llm = GPT4All(model=path, device="cpu", **model_config['gpt4all_config'])
        
    prompt = PromptTemplate.from_template(model_config['prompt'])
        
    return LLMChain(llm=llm, prompt=prompt)

def generate_chains():
    chains = {}
    for file in os.scandir(models_folder_path):
        if file.name.endswith(".gguf"):
            model_name = file.name.split('.gguf')[0]
            
            llm_chain = generate_llm_chain(file.name)
            
            chains[model_name + ":summarize"] = lambda: SummarizeChain(llm_chain=llm_chain)
            #chains[model_name + ":simplify"] = lambda: SimplifyChain(llm=llm_chain)
            #chains[model_name + ":formalize"] = lambda: FormalizeChain(llm=llm_chain)
            chains[model_name + ":headline"] = lambda: HeadlineChain(llm=llm_chain)
            chains[model_name + ":topics"] = lambda: TopicsChain(llm=llm_chain)
            chains[model_name + ":free_prompt"] = lambda: FreePromptChain(llm=llm_chain)
            
    return chains