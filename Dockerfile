FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

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
    wget \
    xvfb \
    unzip \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libegl1-mesa libgles2-mesa ffmpeg

# $ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf


# Setup conda environment
RUN conda init bash && \
    conda create -n libero python=3.8.13 -y && \
    echo "source activate libero" >> ~/.bashrc

# Install dependencies
WORKDIR /opt/ml/libero
COPY requirements.txt /opt/ml/libero/requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install robosuite
RUN pip install tyro

ENV HF_HOME /opt/ml/input/data/huggingface_cache
ENV LIBERO_CONFIG_PATH /opt/ml/libero/config
ENV HYDRA_FULL_ERROR 1

COPY . /opt/ml/libero/
RUN pip install -e .

ENTRYPOINT ["./run.sh"]