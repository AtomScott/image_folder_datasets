FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

USER root
ENV DEBIAN_FRONTEND=noninteractive

################################################################################
# 1. Install PyTorch and CuPy.
#
# They are very heavy, so run them first.
################################################################################

# Install python to install PyTorch and CuPy.
RUN apt-get -q update && apt-get install -qqy python3 python3-pip

# Install PyTorch.
RUN python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html


# Install CuPy.
RUN python3 -m pip install cupy-cuda110

################################################################################
# 2. Install Other Packages.
################################################################################

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

RUN apt-get -q update & apt-get install -qqy \
    sudo wget nano curl nodejs npm

RUN apt-get install -y git 