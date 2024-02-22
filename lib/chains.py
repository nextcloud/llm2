"""Registers all chains based on the models/ directory contents
"""

import os

from formalize import FormalizeChain
from free_prompt import FreePromptChain
from headline import HeadlineChain
from langchain_community.llms import GPT4All
from simplify import SimplifyChain
from summarize import SummarizeChain
from topics import TopicsChain
from langchain.llms.llamacpp import LlamaCpp
dir_path = os.path.dirname(os.path.realpath(__file__))

# Define your configuration as a dictionary
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

def generate_llm(path):
     try:
         llm = LlamaCpp(
                         model_path=path,
                         device=config["llama"]["model_kwargs"]["device"],
                         n_gpu_layers=config["llama"]["n_gpu_layers"],
                         n_ctx=config["llama"]["n_ctx"],
                         max_tokens=config["llama"]["n_ctx"]
         )
         print(f'\033[1;42mUsing:\033[0m\033[1;32m {config["llama"]["model_kwargs"]["device"]}\033[0m', flush=True)
     except Exception as gpu_error:
         try:
             llm = LlamaCpp(model_path=path, device="cpu", max_tokens=4096)
             print("\033[1;43mUsing:\033[0m\033[1;32m CPU\033[0m", flush=True)
         except Exception as cpu_error:
             raise RuntimeError(f"\033[1;31mError:\033[0m Failed to initialize the LLM model on both GPU and CPU.", f"{cpu_error}") from cpu_error
     return llm

def generate_llm_generator(path):
    models[file.name.split(".gguf")[0]] = lambda: generate_llm(path)


models = {}

for file in os.scandir(dir_path + "/../models/"):
    if file.name.endswith(".gguf"):
        generate_llm_generator(file.path)


def generate_chains(model_name, model):
    chains[model_name + ":summary"] = lambda: SummarizeChain(llm=model())
    #chains[model_name + ":simplify"] = lambda: SimplifyChain(llm=model())
    #chains[model_name + ":formalize"] = lambda: FormalizeChain(llm=model())
    chains[model_name + ":headline"] = lambda: HeadlineChain(llm=model())
    chains[model_name + ":topics"] = lambda: TopicsChain(llm=model())
    chains[model_name + ":free_prompt"] = lambda: FreePromptChain(llm=model())


chains = {}
for model_name, model in models.items():
    generate_chains(model_name, model)
