# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
FROM docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt-get update --fix-missing
RUN apt install -y pipx build-essential git vim curl
RUN pipx install poetry

# Download and install FRP client into /usr/local/bin.
RUN set -ex; \
    ARCH=$(uname -m); \
    if [ "$ARCH" = "aarch64" ]; then \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_arm64.tar.gz"; \
    else \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_amd64.tar.gz"; \
    fi; \
    echo "Downloading FRP client from $FRP_URL"; \
    curl -L "$FRP_URL" -o /tmp/frp.tar.gz; \
    tar -C /tmp -xzf /tmp/frp.tar.gz; \
    mv /tmp/frp_0.61.1_linux_* /tmp/frp; \
    cp /tmp/frp/frpc /usr/local/bin/frpc; \
    chmod +x /usr/local/bin/frpc; \
    rm -rf /tmp/frp /tmp/frp.tar.gz

ENV DEBIAN_FRONTEND=dialog
ENV PATH="/root/.local/bin:${PATH}"
ENV CMAKE_ARGS="-DGGML_CUDA=on"

WORKDIR /app

# Install requirements
COPY pyproject.toml .
COPY poetry.lock .
COPY healthcheck.sh .
COPY --chmod=775 start.sh /

RUN poetry install
RUN ln -s /usr/local/cuda/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/

ADD lib /app/lib
ADD models /app/models
ADD default_config /app/default_config

WORKDIR /app/lib
ENTRYPOINT ["/start.sh", "poetry", "run", "python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/llm2
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /app/healthcheck.sh
