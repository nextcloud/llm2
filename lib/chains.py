"""Registers all chains based on the models/ directory contents
"""

import os

from free_prompt import FreePromptChain
from headline import HeadlineChain
from langchain_community.llms import GPT4All
from langchain.chains import LLMChain
from summarize import SummarizeChain
from topics import TopicsChain
from langchain.prompts import PromptTemplate
import json
from nc_py_api.ex_app import persistent_storage

dir_path = os.path.dirname(os.path.realpath(__file__))
models_folder_path = os.path.join(dir_path , "../models/")

def get_model_config(file_name):
    file_name = file_name.split('.gguf')[0]
    if os.path.exists(os.path.join(models_folder_path, file_name + ".json")):
        model_config_path = os.path.join(models_folder_path, file_name + ".json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    elif os.path.exists(os.path.join(persistent_storage(), file_name + ".json")):
        model_config_path = os.path.join(persistent_storage(), file_name + ".json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config_path = os.path.join(dir_path, "../default_config", "config.json")
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

    if not os.path.exists(path):
        path = os.path.join(persistent_storage(), file_name)

    llm = GPT4All(model=path, device="cpu", **model_config['gpt4all_config'])

    prompt = PromptTemplate.from_template(model_config['prompt'])

    return LLMChain(llm=llm, prompt=prompt)


def generate_chains():
    chains = {}
    for file in os.scandir(models_folder_path):
        if file.name.endswith(".gguf"):
            model_name = file.name.split('.gguf')[0]

            llm_chain = lambda: generate_llm_chain(file.name)

            chains[model_name + ":summary"] = lambda: SummarizeChain(llm_chain=llm_chain())
            chains[model_name + ":headline"] = lambda: HeadlineChain(llm_chain=llm_chain())
            chains[model_name + ":topics"] = lambda: TopicsChain(llm_chain=llm_chain())
            chains[model_name + ":free_prompt"] = lambda: FreePromptChain(llm_chain=llm_chain())

    for file in os.scandir(persistent_storage()):
        if file.name.endswith('.gguf'):
            model_name = file.name.split('.gguf')[0]

            llm_chain = lambda: generate_llm_chain(file.name)

            chains[model_name + ":summary"] = lambda: SummarizeChain(llm_chain=llm_chain())
            chains[model_name + ":headline"] = lambda: HeadlineChain(llm_chain=llm_chain())
            chains[model_name + ":topics"] = lambda: TopicsChain(llm_chain=llm_chain())
            chains[model_name + ":free_prompt"] = lambda: FreePromptChain(llm_chain=llm_chain())


    return chains
