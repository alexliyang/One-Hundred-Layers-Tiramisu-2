#/bin/bash

source activate ImageSegmentation
sudo apt-get update
sudo apt-get install -y git
cd ~ && git clone https://github.com/Clcanny/One-Hundred-Layers-Tiramisu.git
cd ~/One-Hundred-Layers-Tiramisu &&\
    python camvid_data_loader.py &&\
    python model-tiramasu-103.py &&\
    mv tiramisu_fc_dense103_model.json tiramisu_fc_dense67_model_12.json &&\
    KERAS_BACKEND=tensorflow python train-tiramisu.py
source deactivate ImageSegmentation
