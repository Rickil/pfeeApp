# Base image with CUDA 11.7.1 and cuDNN 8 on Ubuntu 20.04 for detectron2 according to https://github.com/facebookresearch/detectron2/issues/5008

# Nvidia's official image to download 
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.7
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Add a PPA for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-apt \
    git \
    cmake \
    wget \
    ninja-build \
    libglib2.0-0 \
    libjpeg-dev \
    libgl1-mesa-glx \
    libpng-dev \
    default-jre 
RUN ln -s /usr/lib/python3/dist-packages/apt_pkg.cpython-310-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/apt_pkg.so
# get latest pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
# Install Torch, version from https://pytorch.org/get-started/previous-versions/
RUN pip3 install --upgrade pip cython \
    opencv-python-headless \
    torch==2.0.1+cu117  \
    torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install cython opencv-python-headless \
    && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /root/Gallica_App
COPY requirements.txt .
RUN pip3 install -r requirements.txt 
RUN ln -sf /bin/bash /bin/sh
COPY Gallica_App .