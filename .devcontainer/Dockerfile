FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

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
RUN python3 -m pip install torch torchvision \
    -f https://download.pytorch.org/whl/torch_stable.html


################################################################################
# 2. Install Other Packages.
################################################################################
# COPY requirements.txt /tmp/
# RUN pip3 install -r /tmp/requirements.txt

