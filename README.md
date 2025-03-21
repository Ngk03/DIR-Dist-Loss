# Dist Loss: Enhancing Regression in Few-Shot Region through Distribution Distance Constraint

This repository contains the code for the paper:

[Dist Loss: Enhancing Regression in Few-Shot Region through Distribution Distance Constraint](https://openreview.net/pdf?id=YeSxbRrDRl)

**Authors**: Guangkun Nie, Gongzheng Tang, Shenda Hong  
**Conference**: ICLR 2025  

<p align="center">
    <img src="figures/intro.png" width="1500"> <br>
    A real-world healthcare task of potassium (K<sup>+</sup>) concentration regression from electrocardiogram (ECG).
</p>

## Applying Dist Loss to Other Tasks

The following example demonstrates how Dist Loss can be applied to other regression tasks.

```python
import torch
import numpy as np
from dist_loss import DistLoss
from loss_utils import get_label_distribution, get_batch_theoretical_labels

# Configuration for Dist Loss
bw_method = 0.4  # Bandwidth for kernel density estimation (KDE) to estimate the training label distribution
min_label, max_label = 1, 101  # Defines the possible label value range [min_label, max_label]
step = 1  # Granularity of labels (e.g., 1 year for age prediction, 0.1 mmol/L for K⁺ concentration)

# Simulated training labels (e.g., age values in years)
train_labels = np.random.randint(21, 81, size=10000)  # Shape: (10000,)

# Loss function configuration
loss_type = 'l1'  # Options: ['l1', 'l2', 'focal_l1', 'huber'] or a custom nn.Module loss
distribution_loss_term_weight = 1.0  # Weight of the distribution alignment loss term
regularization_strength = 0.01  # Parameter for fast differentiable sorting, typically set to 0.01

# Training batch size
batch_size = 256  # Larger values improve accuracy of label distribution approximation

# Estimate training label distribution and generate pseudo-labels
label_density = get_label_distribution(train_labels, bw_method, min_label, max_label, step)
pseudo_labels = get_batch_theoretical_labels(label_density, batch_size, min_label, step)
pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.float32).reshape(-1, 1).cuda()

# Initialize the loss function
loss_fn = DistLoss(
    loss_fn=loss_type,
    loss_weight=distribution_loss_term_weight,
    regularization_strength=regularization_strength
)

# Training loop
for batch_idx, (inputs, targets, weights) in enumerate(train_loader):
    outputs = model(inputs)

    if not args.unweighted:
        # Standard Dist Loss calculation
        # This version already enhances performance in few-shot regions by enforcing distribution alignment.
        loss = loss_fn(outputs, targets, pseudo_labels)
    else:
        # Weighted version of Dist Loss
        # This version further emphasizes few-shot regions by explicitly adjusting loss weights based on label frequency.
        loss = loss_fn(outputs, targets, pseudo_labels, weights)

    loss.backward()
```

## Citation
```bibtex
@inproceedings{nie2025dist,
  title     = {Dist Loss: Enhancing Regression in Few-Shot Region through Distribution Distance Constraint},
  author    = {Nie, Guangkun and Tang, Gongzheng and Hong, Shenda},
  booktitle = {Proceedings of the Thirteenth International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=YeSxbRrDRl}
}
```

## Contact
If you have any questions, please feel free to reach out via email (nieguangkun@stu.pku.edu.cn) or by opening an issue on GitHub. We hope you find this useful!

