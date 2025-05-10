# IMDB-WIKI-DIR
## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. __(Optional)__ We have provided required IMDB-WIKI-DIR meta file `imdb_wiki.csv` to set up balanced val/test set in folder `../../datasets`. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Getting Started

### Stage 1. Train the Base Model
To train a vanilla model as the base model:
```bash
python train.py --distribution_loss_term_weight 0 --not_balanced_metric
```

### Stage 2. Train a Model Using RRT + Dist Loss
To train a model with Dist Loss using inverse probability weighting (chosen in the original paper to focus more on the few-shot regions):

```bash
python train.py \
--data_dir <path_to_data_dir> \
--pretrained <path_to_base_model_ckpt> \
--retrain_fc \
--reweight inverse
```

To train a model with Dist Loss without weighting (this approach is also effective for regression in few-shot regions, placing more emphasis on overall performance compared to the previous version):

```bash
python train.py \
--data_dir <path_to_data_dir> \
--pretrained <path_to_base_model_ckpt> \
--retrain_fc \
--reweight none
```

### Evaluate a trained checkpoint

```bash
python train.py [...evaluation model arguments...] --evaluate --resume <path_to_evaluation_ckpt>
```

**Note:** This implementation is inspired by and references [https://github.com/YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression).
