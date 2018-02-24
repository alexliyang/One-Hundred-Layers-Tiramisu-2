#!/bin/bash

declare -a subdirectory=("train" "val" "test")

for i in "${subdirectory[@]}"; do
    python superpixel.py -d "./CamVid/${i}"
done

exit 0
