# Non-Uniform Adversarially Robust Pruning (AutoML 2022)

Repository with the code to repoduce the results with pre-trained checkpoints for paper: [AutoML 2022 "Non-Uniform Adversarially Robust Pruning"](https://intellisec.de/pubs/2022-automl.pdf). In this work, we propose a reinforcement learning based framework to search for non-uniform pruning strategies as the plug-in to further improve adversarially robust pruning by state-of-the-art methods HYDRA ([Sehweg et al.](https://proceedings.neurips.cc/paper/2020/file/e3a72c791a69f87b05ea7742e04430ed-Paper.pdf)) and R-ADMM ([Shaokai Ye et al.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf)).

## Prerequisites
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

## Strategy search
We accomplish the strategy search via running `search_strategy.sh`. There are three phases in this script: 1) download dataset and convert to `.tfrecord`, 2) reload TF models from Pytorch models, and 3) run pruning strategy search. Finally, it yields the file ```Episode_wise_Rewards.csv``` in corresponding folder under root folder `./snapshots`. To prune model with found strategy on Hydra or R-ADMM, the best strategy should be named and saved into `.json` file under folder `./strategies`. 


* Strategy visualization
  
  We use the `pdfplot` package in LaTeX to visualize the averaged strategy distribution with error bar for `Figure 1` and `Figure 3` in the paper. Here, we save our 5 times strategies in folder `./strategies`. To generate each strategy distribution as an PDF file, please move to folder `./stgplots` run `stgplot.sh` with wanted arguments. Note that `pdflatex` is required.


## Pruning with SoTA methods
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

## Pretrain models
To obtain adversarial pretrained models, we reuse Hydra's code and run `hydra_pretrain.sh` in folder `./hydra-nonuniform` such that all pretrained models can be easily reloaded in Hydra and R-ADMM pruning. Note that, we use [FreeAT](https://arxiv.org/pdf/1904.12843.pdf) to robustly train ResNet50 on ImageNet.


## Results summarization
To generate `Table 2` and `Table 3` in our paper, we firstly collect all evaluation results via `gen_summary.sh` to summary all experiment results. Then we save them into `.xls` files in folder `./raw-experiment-results` and convert them to `.csv` files via `xls_2_csv.sh`, such that they can be read by our paper in LaTeX.


---


## Citation

If you find this work helpful, please consider citing it.

```bibtex
@inproceedings{Zhao2022Heracles,
    title={Non-Uniform Adversarially Robust Pruning},
    author={Qi Zhao and Tim Königl and Christian Wressnegger},
    booktitle={Proc. of the International Conference on Automated Machine Learning ({AutoML})},
    year={2022}
}
```
