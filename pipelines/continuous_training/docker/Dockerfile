FROM python:3.12-slim-bookworm
LABEL maintainer="otzslayer@gmail.com"
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG USER_HOME=/home/codespace
ARG UTIL_PATH=utils
ARG PREPROCESSING_PATH=pipelines/continuous_training/data_preprocessing
ARG REQUIREMENTS_PATH=pipelines/continuous_training/docker
ARG TRAINING_PATH=pipelines/continuous_training/training

RUN groupadd --gid 1000 codespace \
    && useradd --uid 1000 --gid codespace --shell /bin/bash --create-home codespace

COPY --chown=codespace:codespace ${UTIL_PATH}/ ${USER_HOME}/utils
COPY --chown=codespace:codespace ${PREPROCESSING_PATH}/preprocessor.py \
    ${USER_HOME}/data_preprocessing/
COPY --chown=codespace:codespace ${TRAINING_PATH}/trainer.py \
    ${USER_HOME}/training/
COPY --chown=codespace:codespace ${REQUIREMENTS_PATH}/requirements.txt \
    ${USER_HOME}/

USER codespace
WORKDIR ${USER_HOME}

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="${USER_HOME}/.venv"

ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN mkdir -p ${USER_HOME}/artifacts \
    && uv init --python 3.12 \
    && uv venv --python 3.12 --seed \
    && uv pip install -r ${USER_HOME}/requirements.txt