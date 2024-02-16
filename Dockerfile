#FROM python:3.11-slim-bookworm
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

COPY requirements.txt /

ADD cs[s] /app/css
ADD im[g] /app/img
ADD j[s] /app/js
ADD l10[n] /app/l10n
ADD li[b] /app/lib
ADD model[s] /app/models

RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11
RUN apt-get install -y python3.11-venv
RUN apt-get install -y python3.11-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN apt-get install -y vim
RUN apt-get install -y cuda-nvcc-11-8
RUN apt-get install -y cuda-toolkit-11-8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN apt-get install -y --no-install-recommends pandoc
RUN apt-get -y clean
RUN rm -rf /var/lib/apt/lists/*

RUN python3.11 -m ensurepip --upgrade
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN \
  python3.11 -m pip install -r requirements.txt && rm -rf ~/.cache && rm requirements.txt
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" python3.11 -m pip install llama-cpp-python --force-reinstall

#RUN python3 -m pip install torch --force-reinstall --no-cache-dir


ENV DEBIAN_FRONTEND=dialog
WORKDIR /app/lib
ENTRYPOINT ["python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/llm2
