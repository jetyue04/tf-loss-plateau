FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install tools useful for VS Code remote
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy your environment
COPY env.yml .

# Install conda env
RUN conda env create -f env.yml

# Activate env automatically
SHELL ["conda", "run", "-n", "emerge", "/bin/bash", "-c"]

# Install Jupyter + kernel (optional but recommended)
RUN pip install ipykernel jupyterlab

# Keep container alive
CMD ["sleep", "infinity"]