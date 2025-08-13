FROM python:3.12-slim-bookworm
LABEL maintainer="otzslayer@gmail.com"
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG USER_HOME=/home/codespace
ARG UTIL_PATH=utils
ARG API_PATH=api

RUN groupadd --gid 1000 codespace \
    && useradd --uid 1000 --gid codespace --shell /bin/bash --create-home codespace

COPY --chown=codespace:codespace ${UTIL_PATH}/ ${USER_HOME}/utils
COPY --chown=codespace:codespace ${API_PATH}/bentofile.yaml ${USER_HOME}/
COPY --chown=codespace:codespace ${API_PATH}/requirements.txt ${USER_HOME}/
COPY --chown=codespace:codespace ${API_PATH}/services.py ${USER_HOME}/
COPY --chown=codespace:codespace ${API_PATH}/src/ ${USER_HOME}/${API_PATH}/src

USER codespace
WORKDIR ${USER_HOME}/${API_PATH}
ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="${USER_HOME}/${API_PATH}/.venv"

ENV PATH="${VIRTUAL_ENV}/bin:${USER_HOME}/.local/bin:${PATH}"

RUN uv init --python 3.12 \
    && uv venv --python 3.12 --seed \
    && uv pip install -r ${USER_HOME}/requirements.txt
