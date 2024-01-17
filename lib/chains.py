from formalize import FormalizeChain
from simplify import SimplifyChain
from summarize import SummarizeChain
from headline import HeadlineChain
from topics import TopicsChain
from free_prompt import FreePromptChain
from langchain_community.llms import GPT4All
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


def generate_llm(path):
    try:
        return GPT4All(model=path, device='gpu')
    except:
        return GPT4All(model=path, device='cpu')


def generate_llm_generator(path):
    models[file.name.split('.gguf')[0]] = lambda: generate_llm(path)


models = {}

for file in os.scandir(dir_path + '/../models/'):
    if file.name.endswith('.gguf'):
        generate_llm_generator(file.path)



def generate_chains(model_name, model):
    chains[model_name + ':summarize'] = lambda: SummarizeChain(llm=model())
    chains[model_name + ':simplify'] = lambda: SimplifyChain(llm=model())
    chains[model_name + ':formalize'] = lambda: FormalizeChain(llm=model())
    chains[model_name + ':headline'] = lambda: HeadlineChain(llm=model())
    chains[model_name + ':topics'] = lambda: TopicsChain(llm=model())
    chains[model_name + ':free_prompt'] = lambda: FreePromptChain(llm=model())


chains = {}
for model_name, model in models.items():
    generate_chains(model_name, model)
