# Base image with CUDA 11.7.1 and cuDNN 8 on Ubuntu 20.04 for detectron2 according to https://github.com/facebookresearch/detectron2/issues/5008
#i tried to use 11.3 but i had errors , need confirmation for that too
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.7

# Nvidia's official image to download 
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
    git \
    cmake \
    wget \
    ninja-build \
    libglib2.0-0 \
    libjpeg-dev \
    libgl1-mesa-glx \
    libpng-dev

# get latest pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN pip3 install --upgrade pip
# Install Torch, version from https://pytorch.org/get-started/previous-versions/
RUN pip3 install torch==2.0.1+cu117  torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
#install cocotools and detectron2
RUN pip3 install cython opencv-python-headless 
RUN pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
# create workdir 
WORKDIR /Gallica_App
COPY requirements.txt .
RUN pip3 install -r requirements.txt 

COPY Gallica_App .