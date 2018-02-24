#!/bin/bash

declare -a sudirectory=("train" "val" "test")

for i in "${subdirectory[@]}"; do
    echo ${i}
    python superpixel.py -d "./CamVid/${i}"
done

exit 0
