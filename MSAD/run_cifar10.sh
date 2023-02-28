#!/bin/bash

DATASET_PATH="/cifar10/"

sudo nvidia-smi -i 0

for (( n=0; n<10; n++ ))
do
  echo "Running MSAD on CIFAR10 with label $n"
  python main.py --dataset=cifar10 --label=$n --backbone=18 --dataset_path=$DATASET_PATH --epochs=5
  sudo nvidia-smi --gpu-reset
done
