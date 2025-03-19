FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

SHELL ["/bin/bash", "--login", "-c"]


RUN apt-get update --allow-unauthenticated && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y wget


# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx libegl1-mesa libgles2-mesa ffmpeg

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
ENV PATH="~/miniconda3/bin:$PATH"

# Setup conda environment
RUN conda init bash && \
    conda create -n libero python=3.8.13 -y && \
    echo "source activate libero" >> ~/.bashrc

# Install dependencies
WORKDIR /opt/ml/libero
COPY requirements.txt /opt/ml/libero/requirements.txt
COPY ppo/requirements-ppo.txt /opt/ml/libero/ppo/requirements-ppo.txt

RUN source ~/miniconda3/bin/activate && conda activate libero && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r ppo/requirements-ppo.txt && \
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install --upgrade imageio imageio[ffmpeg] imageio-ffmpeg && \
    pip install -e .

# RUN pip install -r requirements.txt
# RUN pip install -r ppo/requirements-ppo.txt
# RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install --upgrade imageio imageio[ffmpeg] imageio-ffmpeg
# RUN pip install robosuite

ENV HF_HOME /opt/ml/input/data/huggingface_cache
ENV LIBERO_CONFIG_PATH /opt/ml/libero/config
ENV HYDRA_FULL_ERROR 1

COPY . /opt/ml/libero/
RUN pip install -e .

# ENTRYPOINT ["./run.sh"]
ENTRYPOINT ["/bin/bash", "-c", "source ~/miniconda3/etc/profile.d/conda.sh && conda activate libero && exec ./run.sh"]
