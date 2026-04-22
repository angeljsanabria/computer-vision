# Ubuntu 22.04 + Python 3.10 para RKNN-Toolkit2 (export ONNX -> RKNN en PC x86_64)
# Build (desde Mac Apple Silicon, usar --platform linux/amd64):
#   docker build --platform linux/amd64 -t rknn-toolkit2-export .

FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ca-certificates \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Contexto de build: directorio padre edge_iot_myd (ver README en esta carpeta)
COPY requirements-toolkit-host.txt /workspace/requirements-toolkit-host.txt

RUN python3.10 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-toolkit-host.txt

# El codigo del repo se monta con -v al ejecutar el contenedor
CMD ["/bin/bash"]
