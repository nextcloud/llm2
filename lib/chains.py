"""Registers all chains based on the models/ directory contents
"""

import os

from free_prompt import FreePromptChain
from headline import HeadlineChain
from topics import TopicsChain
from summarize import SummarizeChain
from contextwrite import ContextWriteChain
from reformulate import ReformulateChain
from simplify import SimplifyChain
from formalize import FormalizeChain
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
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


config = {
    "llama": {
        "n_batch": 10,
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "model_kwargs": {
            "device": "cuda"
        }
    }
}


def generate_llm_chain(file_name):
    model_config = get_model_config(file_name)

    path = os.path.join(models_folder_path, file_name)

    if not os.path.exists(path):
        path = os.path.join(persistent_storage(), file_name)

    try:
        llm = LlamaCpp(
            model_path=path,
            model_kwargs={'device': config["llama"]["model_kwargs"]["device"]},
            n_gpu_layers=config["llama"]["n_gpu_layers"],
            n_ctx=model_config['gpt4all_config']["n_predict"],
            max_tokens=model_config["gpt4all_config"]["max_tokens"],
            stop=model_config["gpt4all_config"]["stop"],
            echo=True
        )
        print(f'Using: {config["llama"]["model_kwargs"]["device"]}', flush=True)
    except Exception as gpu_error:
        try:
            llm = LlamaCpp(model_path=path, device="cpu",
                           n_ctx=model_config['gpt4all_config']["n_predict"],
                           max_tokens=model_config["gpt4all_config"]["max_tokens"],
                           stop=model_config["gpt4all_config"]["stop"],
                           echo=True)
            print("Using: CPU", flush=True)
        except Exception as cpu_error:
            raise RuntimeError(f"Error: Failed to initialize the LLM model on both GPU and CPU.", f"{cpu_error}") from cpu_error

    prompt = PromptTemplate.from_template(model_config['prompt'])

    return LLMChain(llm=llm, prompt=prompt)




def generate_chains():
    chains = {}
    for file in os.scandir(models_folder_path):
        if file.name.endswith(".gguf"):
            model_name = file.name.split('.gguf')[0]

            chain = [None]
            llm_chain = lambda:  chain[-1] if chain[-1] is not None else chain.append(generate_llm_chain(file.name)) or chain[-1]

            chains[model_name + ":core:text2text:summary"] = lambda: SummarizeChain(llm_chain=llm_chain())
            chains[model_name + ":core:text2text:headline"] = lambda: HeadlineChain(llm_chain=llm_chain())
            chains[model_name + ":core:text2text:topics"] = lambda: TopicsChain(llm_chain=llm_chain())
            # chains[model_name + ":core:text2text:simplification"] = lambda: SimplifyChain(llm_chain=llm_chain())
            # chains[model_name + ":core:text2text:formalization"] = lambda: FormalizeChain(llm_chain=llm_chain())
            # chains[model_name + ":core:text2text:reformulation"] = lambda: ReformulateChain(llm_chain=llm_chain())
            chains[model_name + ":core:text2text"] = lambda: FreePromptChain(llm_chain=llm_chain())
            chains[model_name + ":core:contextwrite"] = lambda: ContextWriteChain(llm_chain=llm_chain())

    for file in os.scandir(persistent_storage()):
        if file.name.endswith('.gguf'):
            model_name = file.name.split('.gguf')[0]

            chain = [None]
            llm_chain = lambda:  chain[-1] if chain[-1] is not None else chain.append(generate_llm_chain(file.name)) or chain[-1]

            chains[model_name + ":core:text2text:summary"] = lambda: SummarizeChain(llm_chain=llm_chain())
            chains[model_name + ":core:text2text:headline"] = lambda: HeadlineChain(llm_chain=llm_chain())
            chains[model_name + ":core:text2text:topics"] = lambda: TopicsChain(llm_chain=llm_chain())
            # chains[model_name + ":core:text2text:simplification"] = lambda: SimplifyChain(llm_chain=llm_chain())
            # chains[model_name + ":core:text2text:formalization"] = lambda: FormalizeChain(llm_chain=llm_chain())
            # chains[model_name + ":core:text2text:reformulation"] = lambda: ReformulateChain(llm_chain=llm_chain())
            chains[model_name + ":core:text2text"] = lambda: FreePromptChain(llm_chain=llm_chain())
            chains[model_name + ":core:contextwrite"] = lambda: ContextWriteChain(llm_chain=llm_chain())


    return chains
