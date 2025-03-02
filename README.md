# Dist Loss: Enhancing Regression in Few-Shot Region through Distribution Distance Constraint

This repository contains the code for the paper:

[Dist Loss: Enhancing Regression in Few-Shot Region through Distribution Distance Constraint](https://openreview.net/pdf?id=YeSxbRrDRl)

Guangkun Nie, Gongzheng Tang, Shenda Hong

ICLR 2025

![A real-world healthcare task of potassium (K$^+$) concentration regression from ECGs. (a) Both hyperkalemia (high K$^+$) and hypokalemia (low K$^+$) are predominantly found in the few-shot region, with normal K$^+$ concentrations located in the many-shot region. Hyperkalemia and hypokalemia are life-threatening conditions that can lead to cardiac arrest and ventricular fibrillation, necessitating accurate and timely detection. Conversely, normal K$^+$ concentrations (the many-shot region) are of little concern, as inaccurate and untimely detection of these samples has minimal impact. Here, we follow \cite{pmlr-v139-yang21m} to define the few-, median-, many-shot regions. (b) illustrates the significant distribution discrepancy between the vanilla model's predictions and the labels, stemming from the imbalanced data distribution. The term "vanilla model" refers to a model that employs no specialized techniques to address imbalanced data. The orange histogram represents the label distribution, while the blue histogram depicts the prediction distribution from the vanilla model. It is evident that the model's predictions are heavily concentrated in the many-shot region and seldom fall into the few-shot region. (c) demonstrates the effectiveness of Dist Loss in reducing the distribution discrepancy. The orange histogram indicates the label distribution, and the blue histogram shows the prediction distribution from the model enhanced with Dist Loss. The distribution discrepancy is significantly reduced.](figures/intro.png)

## Quick review
