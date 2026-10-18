# Dockerfile.serverless  (v1.1)
# Layers a baked-in model onto YOUR existing llama.cpp CUDA image (the one
# your current entrypoint.sh already runs in), so a Serverless worker never
# downloads weights at start-up.
#
# Build (needs about 45 GB free disk and a fast upload):
#   docker build --platform linux/amd64 \
#     --build-arg BASE_IMAGE=YOURUSER/qwen38-llama:CURRENT_TAG \
#     --build-arg QUANT=Q5_K_M-multilingual-mtp \
#     -t YOURUSER/qwen38-llama-serverless:q5km-mtp-v1 -f Dockerfile.serverless .
#   docker push YOURUSER/qwen38-llama-serverless:q5km-mtp-v1
#
# One image per quant. Suggested tags:
#   24 GB GPUs -> QUANT=Q5_K_M-multilingual-mtp
#   48 GB GPUs -> QUANT=Q8_0-multilingual-mtp

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG QUANT=Q5_K_M-multilingual-mtp
ARG REPO=0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF
ARG MMPROJ=mmproj-Qwen3.8-27B-Q8_0.gguf
ARG WITH_VISION=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Bake the model and the vision projector into the image.
RUN mkdir -p /opt/models \
 && curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 \
      -o "/opt/models/RVN-${QUANT}.gguf" \
      "https://huggingface.co/${REPO}/resolve/main/RVN-${QUANT}.gguf" \
 && if [ "${WITH_VISION}" = "1" ]; then \
      curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 \
        -o "/opt/models/${MMPROJ}" \
        "https://huggingface.co/${REPO}/resolve/main/${MMPROJ}"; \
    fi \
 && ls -lh /opt/models

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY health_shim.py /opt/health_shim.py
# Windows safety net: strip CRLF line endings so the shebang and shell syntax work.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh /opt/health_shim.py \
 && chmod +x /usr/local/bin/entrypoint.sh /opt/health_shim.py

# QUANT_OVERRIDE pins the baked file so the entrypoint never tries to
# auto-select (and download) a different quant on a different GPU.
ENV SERVERLESS=1 \
    LLAMA_CACHE=/opt/models \
    QUANT_OVERRIDE=${QUANT} \
    PORT=8080 \
    PORT_HEALTH=8081 \
    ENABLE_VISION=${WITH_VISION} \
    THINKING=false \
    CONTEXT_SIZE=32768

EXPOSE 8080 8081
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
