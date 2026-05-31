# Base image tracks the latest STABLE SGLang release.
# v0.5.12.post1 pins transformers==5.6.0 / torch==2.11.0 / flashinfer==0.6.11.post1.
# The -cu130 tag is SGLang's default CUDA 13.0 build; its prebuilt kernels cover
# SM80 (Ampere) through SM120 (Blackwell, incl. RTX PRO 6000). Requires host
# NVIDIA driver R580+. For older datacenter drivers (R525/R570), swap to
# lmsysorg/sglang:v0.5.12.post1-cu129 (same arch coverage, larger image).
FROM lmsysorg/sglang:v0.5.12.post1-cu130

# Install uv package manager
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory to the one already used by the base image
WORKDIR /sgl-workspace

# install dependencies
COPY requirements.txt ./
# The CUDA-13 base is Ubuntu 24.04 / Python 3.12, whose system interpreter is
# PEP-668 "externally managed"; --break-system-packages lets uv install into it
# (the old cu126 base on Ubuntu 22.04 did not need this).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -r requirements.txt

# copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Model download script execution
# Ensure this script uses python3 and handles paths correctly relative to /app if needed
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 download_model.py; \
    fi

CMD ["python3", "handler.py"]
