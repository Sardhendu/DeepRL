FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

# Ubuntu packages + Numpy
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        g++  \
        git  \
        curl  \
        cmake \
        zlib1g-dev \
        libjpeg-dev \
        xvfb \
        libav-tools \
        xorg-dev \
        libboost-all-dev \
        libsdl2-dev \
        swig \
        python3  \
        python3-dev  \
        python3-future  \
        python3-pip  \
        python3-setuptools  \
        python3-wheel  \
        python3-tk \
        libopenblas-base  \
        libatlas-dev  \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN pip install --ignore-installed pip

# Install Pytorch
RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

# Install Python packages - Step 1
COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt

# Install OPEN AI GYM environments
RUN git clone https://github.com/openai/gym.git \
&& cd gym \
&& pip install -e '.[atari]' \
&& pip install -e '.[classic_control]' \
&& pip install -e '.[box2d]'

# Command for jupyter themes (comment if you dont use jupyter themes)
RUN jt -t onedork -fs 11 -altp -tfs 12 -nfs 12 -ofs 10 -cellw 95% -T -cursc r

# Add directory
RUN mkdir /ds

ENV DEBIAN_FRONTEND teletype