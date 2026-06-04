# Based on cuda ready nivdia image
FROM nvidia/cuda:12.8.0-base-ubuntu24.04
  
# Update the package manger and install base functions
RUN apt-get update && apt-get install -y wget unzip curl bzip2 git libboost-all-dev build-essential cmake

# Install and setup miniforge
RUN curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
RUN bash Miniforge3-Linux-x86_64.sh -p /miniforge -b
RUN rm Miniforge3-Linux-x86_64.sh
ENV PATH=/miniforge/bin:${PATH}

# Update conda
RUN conda update -y conda
# Install Python 3.12, pytorch and matching torchvision (0.27.0 for torch 2.12)
RUN conda install -y python=3.12 pytorch torchvision -c pytorch && \
    pip install torchvision==0.27.0

# We do not require a conda environment since the container is allready isolated
# However, we switch user to not install pip packges as root
RUN useradd -ms /bin/bash -u 1004 -g 1000 pytc
ENV PATH="/home/pytc/.local/bin:${PATH}"
USER pytc

# Setup and install dependencies and project source
WORKDIR /home/pytc/
RUN chmod -R 777 /home/pytc/
# Setup and install waterz
RUN cd /home/pytc/ && git clone https://github.com/PytorchConnectomics/waterz.git
RUN pip install --upgrade pip setuptools
# RUN pip install --upgrade pip setuptools==59.5.0
RUN cd waterz && pip install --upgrade pip && pip install --editable .
# Setup and install local pytorch_connectomics source from the build context.
# Build from repository root so uncommitted host changes are included.
COPY --chown=pytc:1000 . /home/pytc/pytorch_connectomics
RUN cd /home/pytc/pytorch_connectomics && \
    pip install --upgrade pip && \
    pip install --editable .