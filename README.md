# Non-Uniform Adversarially Robust Pruning

Neural networks often are highly redundant and can thus be effectively compressed to a fraction of their initial size 
using model pruning techniques without harming the overall prediction accuracy. Additionally, pruned networks need to 
maintain robustness against attacks such as adversarial examples. Recent research on combining all these objectives has 
shown significant advances using uniform compression strategies, that is, parameters are compressed equally according 
to a preset compression ratio. With this project, we show that employing non-uniform compression strategies allows to 
improve clean data accuracy as well as adversarial robustness under high overall compression, in particular using 
channel pruning. We leverage reinforcement learning for finding an optimal trade-off and demonstrate that the resulting 
compression strategy can be used as a plug-in replacement for uniform compression ratios of existing state-of-the-art 
approaches.

For further details please consult the [conference publication](https://intellisec.de/pubs/2022-automl.pdf).

<img src="https://intellisec.de/research/heracles/overview.svg" width="1000" /><br />


## Publication
A detailed description of our work is presented at the [(AutoML 2022)](https://2023.automl.cc/) in July 2022. If you 
would like to cite our work, please use the reference as provided below:

```
@InProceedings{Zhao2022Heracles,
author    = {Qi Zhao and Tim Königl and Christian Wressnegger},
title     = {Non-Uniform Adversarially Robust Pruning},
booktitle = {Proc. of the International Conference on Automated
Machine Learning ({AutoML})},
year      = 2022,
month     = jul
}
```

A preprint of the paper is available [here]([https://intellisec.de/pubs/2022-automl.pdf]).

## Code

### Prerequisites
All code is compatible with Tensorflow 1.15. Building environment for code running can be realized by `requirements.tex`. We finish all implementation on Cuda 11. Following commands may help you to build up the needed environment via `conda` and `pip`:
```
conda create -n heracles python=3.6
conda activate heracles
pip install nvidia-pyindex==1.0.8
pip install -r requirements.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Here we offer the remotely stored pretrained PyTorch models as `.zip` file for the following experiments. To obtain them, please run:
```
bash ./Pretrain-Models/download_pretrain.sh
```

In our following `.sh` scripts, we define arguments as below, which can be changed for the experiment with different network and dataset. 
   
| ARCH   | DATASET | P_Rate       | P_CH_Rate               | P_W_Rate               |P_Reg         | STG_ID      | GPU    |
|--------|---------|--------------|-------------------------|------------------------|--------------|-------------|--------|
|network | dataset | pruning rate |channel P_Rate (for plot)|weight P_Rate (for plot)| pruning mode | strategy id | gpu id |

### Strategy search
We accomplish the strategy search via running `search_strategy.sh`. There are three phases in this script: 1) download dataset and convert to `.tfrecord`, 2) reload TF models from Pytorch models, and 3) run pruning strategy search. Finally, it yields the file ```Episode_wise_Rewards.csv``` in corresponding folder under root folder `./snapshots`. To prune model with found strategy on Hydra or R-ADMM, the best strategy should be named and saved into `.json` file under folder `./strategies`. 


* Strategy visualization
  
  We use the `pdfplot` package in LaTeX to visualize the averaged strategy distribution with error bar for `Figure 1` and `Figure 3` in the paper. Here, we save our 5 times strategies in folder `./strategies`. To generate each strategy distribution as an PDF file, please move to folder `./stgplots` run `stgplot.sh` with wanted arguments. Note that `pdflatex` is required.


### Pruning with SoTA methods
Further pruning with `HYDRA` and `R-ADMM` can be accomplished in their corresponding folders `./hydra-nonuniform` and `./r-admm-nonuniform`.

1. Hydra pruning
   ```
   cd hydra-nonuniform
   bash hydra_prune.sh
   ``` 
   
2. R-ADMM pruning
   Select your wanted dataset: $DATASET = ['cifar10', 'svhn', 'imagenet'], then run:
   ```
   cd r-admm-nonuniform/ADMM_examples/$DATASET
   bash radmm-prune.sh
   ```

---

### Pretrain models
To obtain adversarial pretrained models, we reuse Hydra's code and run `hydra_pretrain.sh` in folder `./hydra-nonuniform` such that all pretrained models can be easily reloaded in Hydra and R-ADMM pruning. Note that, we use [FreeAT](https://arxiv.org/pdf/1904.12843.pdf) to robustly train ResNet50 on ImageNet.


### Results summarization
To generate `Table 2` and `Table 3` in our paper, we firstly collect all evaluation results via `gen_summary.sh` to summary all experiment results. Then we save them into `.xls` files in folder `./raw-experiment-results` and convert them to `.csv` files via `xls_2_csv.sh`, such that they can be read by our paper in LaTeX.

