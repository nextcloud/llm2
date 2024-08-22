"""Registers all chains based on the models/ directory contents
"""

import os
import json

from free_prompt import FreePromptChain
from headline import HeadlineChain
from topics import TopicsChain
from summarize import SummarizeChain
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
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

    is_gpu = os.getenv("COMPUTE_DEVICE", "cpu") != "cpu"
    llm = LlamaCpp(
        model_path=path,
        **{
            "n_gpu_layers": (0, -1)[is_gpu],
            **model_config["loader_config"],
        },
    )
    prompt = PromptTemplate.from_template(model_config['prompt'])

    return LLMChain(llm=llm, prompt=prompt)


def generate_chains():
    chains = {}
    for file in os.scandir(models_folder_path):
        if file.name.endswith(".gguf"):

            generate_chain_for_model(file.name, chains)

    for file in os.scandir(persistent_storage()):
        if file.name.endswith('.gguf'):
            generate_chain_for_model(file.name, chains)

    return chains


def generate_chain_for_model(file_name, chains):
    model_name = file_name.split('.gguf')[0]

    chain = [None]
    llm_chain = lambda:  chain[-1] if chain[-1] is not None else chain.append(generate_llm_chain(file_name)) or chain[-1]

    chains[model_name + ":core:text2text:summary"] = lambda: SummarizeChain(llm_chain=llm_chain())
    chains[model_name + ":core:text2text:headline"] = lambda: HeadlineChain(llm_chain=llm_chain())
    chains[model_name + ":core:text2text:topics"] = lambda: TopicsChain(llm_chain=llm_chain())
    # chains[model_name + ":core:text2text:simplification"] = lambda: SimplifyChain(llm_chain=llm_chain())
    # chains[model_name + ":core:text2text:formalization"] = lambda: FormalizeChain(llm_chain=llm_chain())
    # chains[model_name + ":core:text2text:reformulation"] = lambda: ReformulateChain(llm_chain=llm_chain())
    chains[model_name + ":core:text2text"] = lambda: FreePromptChain(llm_chain=llm_chain())
    #chains[model_name + ":core:contextwrite"] = lambda: ContextWriteChain(llm_chain=llm_chain())