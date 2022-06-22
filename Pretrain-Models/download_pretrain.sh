#!/bin/bash

echo 'Downloading pretrained models ...'
wget https://www.dropbox.com/s/u0wavt1kw827nsj/torch-models.zip -P ./Pretrain-Models
echo 'Unzip pretrained models ...'
unzip Pretrain-Models/torch-models.zip -d ./Pretrain-Models
