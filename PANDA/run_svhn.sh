#!/bin/bash

DATASET_PATH="/SVHN/"

nvidia-smi -i 0

for (( n=0; n<10; n++ ))
do
  echo "Running Panda on SVHN with label $n"
  python panda.py --dataset=svhn --label=$n --dataset_path=$DATASET_PATH --ewc
  nvidia-smi --gpu-reset
done
