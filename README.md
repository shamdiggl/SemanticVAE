# VAE with Semantic Consistency Loss on MNIST

This project explores enhancing the interpretability of Variational Autoencoder (VAE) latent spaces. The goal is to train a VAE where specific directions in the latent space correspond to consistent semantic changes in the generated image, regardless of the starting point. This is achieved by introducing a novel "semantic variance" loss term during training.

The project, for the time being, compares three models trained on the MNIST dataset:
1.  **Standard VAE:** A baseline ResNet-based VAE.
2.  **β-VAE:** A VAE trained with β=5 (indicated by filename) to encourage disentanglement.
3.  **Semantic VAE:** Our proposed VAE trained with the additional semantic variance loss term.

## Overview

Standard VAEs often learn complex, entangled latent representations. While they can reconstruct and generate data, understanding *what* each latent dimension controls can be difficult. This project aims for a "linearly decomposed" latent space with respect to semantics.

The core idea is that moving a latent code `z` by a vector `d` should result in a change in the output image that has a *consistent semantic meaning* across different starting points `z`. We measure this consistency using features extracted from a pre-trained classifier and penalize the variance of the semantic difference.

## Key Features

*   ResNet-based VAE architecture (`ResNet_AutoEncoder` in `models.py`).
*   ResNet-based Classifier (`Classifier` in `models.py`) used for feature extraction.
*   Implementation and evaluation of a standard VAE, β-VAE (β=5), and a VAE with a semantic consistency objective.
*   Evaluation script (`Evaluation.py`) comparing models on:
    *   Reconstruction Quality (MSE).
    *   Custom Interpretability Metric ("Global Semantic Variance").
    *   Qualitative Latent Space Interpolation.
    *   Distribution of Latent Code Norms.

## Directory structure

├── models.py             # Contains VAE and Classifier model definitions  
├── Evaluation.py         # Script to load trained models and evaluate them   
├── weights/              # Directory containing pre-trained model weights  
│   ├── classifier.pth  
│   ├── VAE_epoch50.pth  
│   ├── b5-VAE_epoch50.pth  
│   └── semanticVAE_epoch50.pth  
├── data/                 # MNIST dataset storage  
├── README.md             
  
