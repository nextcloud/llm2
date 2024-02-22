FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    software-properties-common

COPY requirements.txt /

ADD li[b] /app/lib
ADD model[s] /app/models

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11
RUN apt-get install -y python3.11-venv
RUN apt-get install -y python3.11-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y libvulkan1
RUN apt-get install -y libnvidia-gl-535-server
RUN apt-get install -y cuda-nvcc-12-2
RUN apt-get install -y cuda-toolkit-12-2
RUN apt-get install -y nvidia-utils-535
RUN apt-get install -y git vim wget curl strace less iputils-ping telnet

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
