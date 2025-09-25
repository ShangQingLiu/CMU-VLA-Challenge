FROM ros:noetic-ros-core-focal
ENV DEBIAN_FRONTEND=noninteractive

# --- System + ROS tooling ---
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    vim \
    python3-catkin-tools python3-osrf-pycommon git cmake \
    libusb-dev python-yaml python-is-python3 \
    curl ca-certificates bzip2 \
    ros-noetic-diagnostic-updater \
    ros-noetic-roslint \
    ros-noetic-cv-bridge \
    ros-noetic-rviz \
    ros-noetic-diagnostic-aggregator \
    ros-noetic-compressed-image-transport \
    ros-noetic-rviz-plugin-tutorials \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-image-geometry \
    ros-noetic-rqt-graph \
    python3.8 python3.8-venv python3.8-distutils \
    python3-rosdep python3-rosinstall python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# pip for Python 3.8 (3.8-specific bootstrap)
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o /tmp/get-pip.py \
 && python3.8 /tmp/get-pip.py \
 && ln -sf /usr/local/bin/pip3.8 /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip3.8 /usr/local/bin/pip3

# --- rosdep BEFORE conda ---
RUN rosdep init && rosdep update --rosdistro $ROS_DISTRO

# ROS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*


# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /opt/conda && \
    rm -f miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

# Source ROS on login shells
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /opt/conda/bin/activate" >> /root/.bashrc


COPY environment.yml /home/environment.yml
COPY requirements.txt /home/requirements.txt

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -f /home/environment.yml

RUN exec bash 
# Activate conda environment
RUN conda init \
&& exec bash 

# --- clone and install habitat-lab v0.1.7 ---
WORKDIR /home
RUN git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git

WORKDIR /home/habitat-lab
# Remove TF 1.x line so pip won’t try to install it

RUN sed -i '/^tensorflow==1\.13\.1$/d' habitat_baselines/rl/requirements.txt
RUN conda init \
 && exec bash \
 && conda activate TAMSxGalbot \
 && python -m pip install --upgrade pip \
 && pip install numpy==2.2.0 \ 
 && python -m pip install --no-cache-dir -r habitat_baselines/rl/requirements.txt \
 && python -m pip install --no-cache-dir -r habitat_baselines/rl/ddppo/requirements.txt 

# If setup.py expects to run from repo root, keep WORKDIR here
#RUN python -m pip install  --use-pep517 -e .
RUN conda init \
 && exec bash \
 && conda activate TAMSxGalbot \
 && python setup.py develop --all 

# --- LLaVA v1.2.2.post1 ---
WORKDIR /home
RUN git clone  --branch v1.2.2.post1 https://github.com/haotian-liu/LLaVA.git

WORKDIR /home/LLaVA
# Use modern editable install; no build isolation so it reuses your preinstalled torch/cuda stack
RUN conda init \
 && exec bash \ 
 && conda activate TAMSxGalbot \
 && python -m pip install  -e . 


