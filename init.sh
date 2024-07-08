#! /bin/bash
export PATH="/root/.local/bin/:$PATH"
export CMAKE_ARGS="-DLLAMA_CUDA=on"
source $(poetry env info --path)/bin/activate
poetry install
cd lib
poetry run python3 main.py