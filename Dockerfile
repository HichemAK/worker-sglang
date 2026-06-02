# Base image bumped from the 1.2.0 default (v0.4.6.post4-cu124) to the latest
# STABLE SGLang. v0.5.12.post1 pins transformers==5.6.0 / torch==2.11.0 /
# flashinfer==0.6.11.post1, so recent architectures (e.g. qwen3_5) parse instead
# of failing config validation. The -cu129 (CUDA 12.9) build's prebuilt kernels
# cover SM80 (Ampere) through SM120 (Blackwell, incl. RTX PRO 6000) and run on
# any CUDA-12 host driver (R525+) via minor-version compatibility -- including
# RunPod's current fleet (observed: RTX 4090 on driver 565 / CUDA 12.7). The
# cu130 default needs R580+ and fails CUDA init on those hosts.
FROM lmsysorg/sglang:v0.5.12.post1-cu129

# The cu129 base bakes NVIDIA_REQUIRE_CUDA="cuda>=12.9", which makes the NVIDIA
# container runtime REFUSE to start the container (at init, before any code runs)
# on hosts whose driver reports CUDA < 12.9 -- e.g. RunPod's RTX 4090 fleet on
# driver 565 / CUDA 12.7:
#   nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.9
# That gate is keyed to the toolkit's *native* driver and ignores CUDA-12
# minor-version compatibility. Relax it to the CUDA-12 floor (R525 / CUDA 12.0)
# so the container starts; the cu129 toolkit then runs on any R525+ driver via
# minor-version compatibility. (It still won't run on CUDA-11 drivers.)
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.0"

# Install uv package manager (-f so it is idempotent if the base already ships uv)
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory to the one already used by the base image
WORKDIR /sgl-workspace

# install dependencies
COPY requirements.txt ./
# --break-system-packages: the v0.5.12 base is Ubuntu 24.04 / Python 3.12, whose
# system interpreter is PEP-668 "externally managed" (the old cu124 base was not).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -r requirements.txt

# copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_PATH=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_PATH=$MODEL_PATH \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_XET_HIGH_PERFORMANCE=1

# Model download script execution
# Ensure this script uses python3 and handles paths correctly relative to /app if needed
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_PATH" ]; then \
        python3 download_model.py; \
    fi

CMD ["python3", "handler.py"]
