#!/bin/bash
ARCH="vgg16_bn" # vgg16_bn, wrn_28_4, resnet18, ResNet50 (only for imagenet)
DATASET="CIFAR10"  # CIFAR10, SVHN, imagenet (only for resnet50)
P_CH_Rate=0.5  # prune rate for channel pruning
P_W_Rate=0.1  # prune rate for weight pruning

echo "Visualize strategy"

python stg_4_tikz.py --arch $ARCH --dataset $DATASET --prune_reg 'channel' --p_rate $P_CH_Rate
python stg_4_tikz.py --arch $ARCH --dataset $DATASET --prune_reg 'weight' --p_rate $P_W_Rate

pdflatex -output-directory="./$DATASET" "\newcommand\dataset{$DATASET}\input{$ARCH.tex}"
