#!/bin/bash
ARCH="vgg16_bn" # vgg16_bn, wrn_28_4, resnet18, resnet50 (only for imagenet)
DATASET="CIFAR10"  # CIFAR10, SVHN, imagenet (only for resnet50)
P_Rate=0.1
P_Reg='weight'  # weight, channel
GPU="0"

echo "Download dataset"
# Only support small scale datasets, imagenet download needs querying its official webpage
if [ $DATASET == 'CIFAR10' ]
then
  python ./dataset/tf_convert/_cifar10_2_tfrecord.py
elif [ $DATASET == 'SVHN' ]
then
  python ./dataset/tf_convert/_svhn_2_tfrecord.py
fi

echo "Reload and transform to TF models"
python reload_torch.py --arch $ARCH --dataset $DATASET --gpu_id $GPU

echo "Heracles pruning strategy search"
if [ $P_Reg == 'weight' ]
then
  if [ $P_Rate == 0.01 ]
  then
    Lb=0.01; Ub=0.8
  else
    Lb=0.01; Ub=0.8
  fi
else
  if [ $P_Rate == 1.0 ]
  then
    Lb=0.1; Ub=1.0
  else
    Lb=0.05; Ub=0.5
  fi
fi

python rl_prune.py --arch $ARCH --dataset $DATASET --gpu_id $GPU --target_sparsity $P_Rate --prune_reg $P_Reg --actor_lbound $Lb --actor_ubound $Ub
