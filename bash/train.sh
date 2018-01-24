#/bin/bash

source activate ImageSegmentation
cd One-Hundred-Layers-Tiramisu &&\
    python camvid_data_loader.py &&\
    python model-tiramasu-103.py &&\
    KERAS_BACKEND=tensorflow python train-tiramisu.py
source deactivate ImageSegmentation
