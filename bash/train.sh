#/bin/bash

source activate ImageSegmentation
sudo apt-get update
sudo apt-get install -y git
cd ~ && git clone https://github.com/Clcanny/One-Hundred-Layers-Tiramisu.git
cp -r ~/CamVid ~/One-Hundred-Layers-Tiramisu
cd ~/One-Hundred-Layers-Tiramisu &&\
    python camvid_data_loader.py &&\
    python model-tiramasu-103.py &&\
    KERAS_BACKEND=tensorflow python train-tiramisu.py
source deactivate ImageSegmentation
