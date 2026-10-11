# navdelta: one image for the whole pipeline (shared stages, all four approaches, open-source
# models, viewer). Runs on Linux and on Windows (Docker Desktop, WSL 2 backend), with an
# NVIDIA GPU (--gpus all) or on the CPU. Built once per default approach:
#   nav-delta-voxel_diff, nav-delta-scene_graph, nav-delta-loop_outliers, nav-delta-rgbd_fusion
#   (see docker-compose.yml)
FROM python:3.12.3-slim-bookworm

# Open3D needs these shared libraries even without a display.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libegl1 libgomp1 libglib2.0-0 libx11-6 libxext6 \
      libxrender1 libsm6 \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    NAVDELTA_DATA=/data \
    NAVDELTA_OUTPUTS=/outputs \
    NAVDELTA_MODEL_CACHE=/models

# PyTorch with CUDA 12.6 runtime libraries (also runs on the CPU when no GPU is passed in).
RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# Open3D also loads libusb (RealSense support) when it is imported.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libusb-1.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY navdelta ./navdelta
RUN pip install --no-deps .

# The approach this image runs by default (navdelta run --approach overrides it).
ARG APPROACH=voxel_diff
ENV NAVDELTA_APPROACH=${APPROACH}

VOLUME ["/data", "/outputs", "/models"]
ENTRYPOINT ["navdelta"]
CMD ["run"]
