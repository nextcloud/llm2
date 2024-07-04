FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    software-properties-common

COPY requirements.txt /

ADD cs[s] /app/css
ADD im[g] /app/img
ADD j[s] /app/js
ADD l10[n] /app/l10n
ADD li[b] /app/lib
ADD model[s] /app/models
ADD default_confi[g] /app/default_config

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11
RUN apt-get install -y python3.11-venv
RUN apt-get install -y python3.11-dev
RUN apt-get install -y python3-pip

ENV CMAKE_ARGS="-DLLAMA_CUDA=on"

RUN \
  python3 -m pip install -r requirements.txt && rm -rf ~/.cache && rm requirements.txt

WORKDIR /app/lib
ENTRYPOINT ["python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/llm2
