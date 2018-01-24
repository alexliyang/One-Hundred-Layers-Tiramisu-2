#/bin/bash

source activate ImageSegmentation

# update/install NVIDIA drivers
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-384

# install CUDA
sudo dpkg -i ~/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# test CUDA
cd /usr/local/cuda-9.1/samples/5_Simulations/nbody && sudo make
# ./nbody

# install cuDNN
tar -xzvf ~/cudnn-9.1-linux-x64-v7.tgz
sudo cp -r ~/cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -r ~/cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

conda install -y tensorflow-gpu
conda install -y keras
conda install -y opencv

source deactivate ImageSegmentation
