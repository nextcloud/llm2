FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip vim git pciutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN apt-get -y clean
RUN rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute
ENV DEBIAN_FRONTEND=dialog

ADD li[b] /app/lib
ADD model[s] /app/models
ADD default_confi[g] /app/default_config

# Install requirements
COPY requirements.txt /
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3.11 -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
# RUN python3 -m pip install -vvv --no-cache-dir https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.81-cu124/llama_cpp_python-0.2.81-cp311-cp311-linux_x86_64.whl
# https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.81-cu122/llama_cpp_python-0.2.81-cp311-cp311-linux_x86_64.whl
RUN sed -i '/llama_cpp_python/d' requirements.txt
RUN python3.11 -m pip install --no-cache-dir --no-deps -r requirements.txt

WORKDIR /app/lib
ENTRYPOINT ["python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/llm2
