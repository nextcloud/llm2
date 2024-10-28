# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registers all chains based on the models/ directory contents
"""

import json
from math import ceil
import os

from free_prompt import FreePromptChain
from headline import HeadlineChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from nc_py_api.ex_app import persistent_storage
from summarize import SummarizeChain
from topics import TopicsChain

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

    compute_device = os.getenv("COMPUTE_DEVICE", "CUDA")
    try:
        llm = LlamaCpp(
            model_path=path,
            **{
                "n_gpu_layers": (0, -1)[compute_device != "CPU"],
                **model_config["loader_config"],
            },
        )
    except Exception as e:
        print(f"Failed to load model '{path}' with compute device '{compute_device}'")
        raise e

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
    n_ctx = get_model_config(file_name)["loader_config"]["n_ctx"]
    chunk_size = int(ceil(0.7 * n_ctx))

    chain = [None]
    llm_chain = lambda:  chain[-1] if chain[-1] is not None else chain.append(generate_llm_chain(file_name)) or chain[-1]

    chains[model_name + ":core:text2text:summary"] = lambda: SummarizeChain(llm_chain=llm_chain(), chunk_size=chunk_size)
    chains[model_name + ":core:text2text:headline"] = lambda: HeadlineChain(llm_chain=llm_chain())
    chains[model_name + ":core:text2text:topics"] = lambda: TopicsChain(llm_chain=llm_chain())
    # chains[model_name + ":core:text2text:simplification"] = lambda: SimplifyChain(llm_chain=llm_chain(), chunk_size=chunk_size)
    # chains[model_name + ":core:text2text:formalization"] = lambda: FormalizeChain(llm_chain=llm_chain(), chunk_size=chunk_size)
    # chains[model_name + ":core:text2text:reformulation"] = lambda: ReformulateChain(llm_chain=llm_chain(), chunk_size=chunk_size)
    chains[model_name + ":core:text2text"] = lambda: FreePromptChain(llm_chain=llm_chain())
    # chains[model_name + ":core:contextwrite"] = lambda: ContextWriteChain(llm_chain=llm_chain())
