FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y pipx build-essential
RUN pipx install poetry

ENV DEBIAN_FRONTEND=dialog

ADD li[b] /app/lib
ADD model[s] /app/models
ADD default_confi[g] /app/default_config

# Install requirements
COPY pyproject.toml /app
COPY poetry.lock /app
COPY init.sh /app
COPY healthcheck.sh /app

WORKDIR /app
ENTRYPOINT ["bash", "init.sh"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/llm2
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /app/healthcheck.sh