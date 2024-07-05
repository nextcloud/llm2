#! /bin/bash
PATH="/root/.local/bin/:$PATH"
CMAKE_ARGS="-DLLAMA_CUDA=on"
source $(poetry env info --path)/bin/activate
poetry install
cd lib
poetry run python3 main.py