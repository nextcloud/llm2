FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y pipx build-essential git vim
RUN pipx install poetry

ENV DEBIAN_FRONTEND=dialog
ENV PATH="/root/.local/bin:${PATH}"
ENV CMAKE_ARGS="-DGGML_CUDA=on"

WORKDIR /app

# Install requirements
COPY pyproject.toml .
COPY poetry.lock .
COPY healthcheck.sh .

RUN poetry install
RUN ln -s /usr/local/cuda/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/

ADD li[b] /app/lib
ADD model[s] /app/models
ADD default_confi[g] /app/default_config

WORKDIR /app/lib
ENTRYPOINT ["poetry", "run", "python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/llm2
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /app/healthcheck.sh
