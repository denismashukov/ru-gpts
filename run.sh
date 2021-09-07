#!/bin/bash

# 1) instance for medium g4dn.xlarge with GPU
# https://docs.amazonaws.cn/en_us/AmazonECS/latest/developerguide/ecs-gpu.html
# https://aws.amazon.com/ru/ec2/instance-types/g4/
sudo apt update
sudo apt install -y git unzip wget
sudo apt install -y python3-dev python3-pip python3-setuptools

# 2) Install NVIDIA
# ruGPT-3 work with Torch v1.7.1 https://pytorch.org/get-started/previous-versions/
# For this version we need to install CUDA 11.0
# Install CUDA Toolkit 11.0 https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal
sudo apt install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

export CUDA_HOME=/usr/local/cuda-11.0/

sudo nvidia-smi

# 2) Install AWS CLI https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# sudo ./aws/install
# 3) Install NVIDIA gaming driver https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-gaming-driver
# 4) Install nvidia-docker2 if you use docker container https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 6) Install Torch v1.7.1 https://pytorch.org/get-started/previous-versions/
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# 7) Install dependencies for ru-gpt
pip3 -q install pip --upgrade
pip3 install jupyter numpy pandas tensorboard transformers==3.5.0 deepspeed
pip3 install tensorflow
pip3 install sklearn
pip3 install nltk

# 8) Install apex https://github.com/NVIDIA/apex
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ..
# 9) Download ru-gpt repo
git clone  https://github.com/sberbank-ai/ru-gpts

