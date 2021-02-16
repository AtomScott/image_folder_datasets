FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

WORKDIR /


####################
# Python Environment
####################

# Install Python and its tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools

RUN pip3 -q install pip --upgrade

# copy the dependencies file to the working directory
COPY requirements.txt .
# install dependencies
RUN pip install -r requirements.txt

# nbdev environment
RUN pip install nbdev jupyter jupyterlab


####################
# Git Environment
####################

RUN apt-get install git