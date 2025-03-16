# AgeDB-DIR

## Getting Started

#### Stage 1. Train the base model
Train a vanilla model as the base model: 
```bash
python train.py --distribution_loss_term_weight 0
```

#### Stage 2. Train a model using RRT + Dist Loss
Train a model with Dist Loss (inverse probability weighting / no weighting)
```bash
python train.py \
--data_dir <path_to_data_dir> \
--pretrained <path_to_base_model_ckpt> \
--reweight inverse <em>OR</em> --reweight none \
