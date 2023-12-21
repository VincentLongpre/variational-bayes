# Exploring the latent space of Variational Models

By [Gabriel Missael Barco](https://github.com/GabrielMissael), Michael Matesic, Parker Levesque, and [Vincent Longpre](https://github.com/VincentLongpre).

Final project, IFT 6269 : Probabilistic Graphical Models - Fall 2023, Prof. [Simon Lacoste-Julien](https://www.iro.umontreal.ca/~slacoste/). *Université de Montréal*

## Abstract 📄
The Adversarial Variational Bayes (AVB) method is a modification of the Variational Autoencoder (VAE) generative model, combining elements from Generative Adversarial Networks (GANs) to improve upon the quality of the generated samples from the model. The neural network is used as a black-box approximate inference model, allowing an arbitrarily expressive modelling of the true posterior distribution. We explore this connection between VAEs and GANs by training both a regular VAE and an AVB model on two datasets of differing complexity.
 
## Models 🧠

### Variational Autoencoder (VAE) 🤖

### Adversarial Variational Bayes (AVAE) 🧠

## Datasets 📊

For this project, we will train these models on the datasets used in the original AVB paper namely 4Frames and binarized MNIST. 

### 4Frames
The 4Frames is a simple synthetic dataset composed of the four 2x2 binary matrices shown below

![4Frames Training Examples](relative/path/in/repository/to/image.svg)

### Binarized MNIST

## Code 🖥️

Project structure (directories):
```
.
└── variational-bayes
    ├── VariNet
    │   ├── models
    │   ├── plots
    │   └── utils
    ├── data
    ├── figures
    └── notebooks
```

## Experiments & Results 📈

## Conclusions 📝

## References
1. Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings.
2. Mescheder, L. M., Nowozin, S., & Geiger, A. (2017). Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks. CoRR, abs/1701.04722. Retrieved from http://arxiv.org/abs/1701.04722
