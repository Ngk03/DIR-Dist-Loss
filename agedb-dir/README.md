# AgeDB-DIR

## Installation

#### Prerequisites

1. Download the AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of the AgeDB dataset for the zip password). Extract the contents to the folder `'../../datasets`.

2. __(Optional)__ We provide a pre-made AgeDB-DIR meta file, `agedb.csv`, for setting up a balanced validation/test set in the `./data` folder. To reproduce the results from the paper, please use this file directly. If you prefer to try different balanced splits, you can generate the meta file using the following commands:

    ```bash
    python data/create_agedb.py
    python data/preprocess_agedb.py
    ```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL

## Code Overview

#### Main Files

- `train.py`: Main training and evaluation script
- `create_agedb.py`: Script to create raw AgeDB meta data
- `preprocess_agedb.py`: Script to create the AgeDB-DIR meta file `agedb.csv` with a balanced validation/test set

#### Main Arguments

- `--data_dir`: Directory to place data and meta files
- `--reweight`: Cost-sensitive re-weighting scheme to use
- `--retrain_fc`: Whether to retrain the regressor
- `--loss`: Type of training loss to use
- `--resume`: Path to resume from a checkpoint (for both training and evaluation)
- `--evaluate`: Flag to evaluate only
- `--pretrained`: Path to load backbone weights for regressor re-training (RRT)

## Getting Started

### Stage 1. Train the Base Model
To train a vanilla model as the base model:
```bash
python train.py --distribution_loss_term_weight 0
```

### Stage 2. Train a Model Using RRT + Dist Loss
To train a model with Dist Loss using inverse probability weighting (chosen in the original paper to focus more on the few-shot regions):

```bash
python train.py \
--data_dir <path_to_data_dir> \
--pretrained <path_to_base_model_ckpt> \
--retrain_fc \
--reweight inverse

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
