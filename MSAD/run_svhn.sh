#!/bin/bash

DATASET_PATH="/SVHN/"

sudo nvidia-smi -i 0

for (( n=0; n<10; n++ ))
do
  echo "Running MSAD on SVHN with label $n"
  python main.py --dataset=svhn --label=$n --backbone=18 --dataset_path=$DATASET_PATH --epochs=5
  sudo nvidia-smi --gpu-reset
done
