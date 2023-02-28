#!/bin/bash

DATASET_PATH="/Fashion-MNIST/"

sudo nvidia-smi -i 0

for (( n=0; n<10; n++ ))
do
  echo "Running MSAD on Fashion-MNIST with label $n"
  python main.py --dataset=fashion --label=$n --backbone=18 --dataset_path=$DATASET_PATH --epochs=5
  sudo nvidia-smi --gpu-reset
done
