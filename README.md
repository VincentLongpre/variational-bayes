# Exploring the latent space of Variational Models

By [Gabriel Missael Barco](https://github.com/GabrielMissael), Michael Matesic, Parker Levesque, and [Vincent Longpre](https://github.com/VincentLongpre).

Final project, IFT 6269 : Probabilistic Graphical Models - Fall 2023, Prof. [Simon Lacoste-Julien](https://www.iro.umontreal.ca/~slacoste/). *Université de Montréal*

## Abstract 📄
The Adversarial Variational Bayes (AVB) method is a modification of the Variational Autoencoder (VAE) generative model, combining elements from Generative Adversarial Networks (GANs) to improve upon the quality of the generated samples from the model. The neural network is used as a black-box approximate inference model, allowing an arbitrarily expressive modelling of the true posterior distribution. We explore this connection between VAEs and GANs by training both a regular VAE and an AVB model on two datasets of differing complexity.
 
## Models 🧠

### Variational Autoencoder (VAE) 🤖

The core concept of VAE is to merge factor analysis's idea of uncovering underlying features with the power of neural network. Using amortized variational inference and the reparametrization trick, this model allows for efficient and scalable learning of complex data distributions.

### Adversarial Variational Bayes (AVB) 🥊

This model introduces adversarial training in the context of variational inference, enhancing the richness and fidelity of generated samples while maintaining the probabilistic framework of VAEs. Adversarial training within this framework aims to minimize the discrepancy between the learned approximate posterior and the true data distribution replacing the KL divergence used in the ELBO loss of VAE.

## Datasets 📊

For this project, we will train these models on the datasets used in the original AVB paper namely 4Frames and binarized MNIST. 

### 4Frames
The 4Frames is a simple synthetic dataset composed of the four 2x2 binary matrices shown below

<div align="center">
    <img src="/figures/4Frames_dataset.png" alt="$Frames Example">
</div>


### Binarized MNIST

The MNIST is composed of 70,000 images of handwritten digits and is frequently used to train computer vision and generative models. For this implementation, we convert the grayscale pixel values into a binary format by setting pixels above a set threshold of 127 to 1 and the rest to 0.

<div align="center">
    <img src="/figures/mnist_dataset.png" alt="$MNIST Example">
</div>

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
    └── trained_models
```

## Experiments & Results 📈

For both datasets, we performed parameter selection and trained comparable architectures for both VAE and AVB models. The findings indicte that AVB seems to allow for more flexibility in samples and reconstructions but for generates less convincing samples due to the greater diversity. Moreover, the difference in the learned latence space is much clearer for dataset with simple structure like 4Frames but becomes hard to visualize once the complexity of the data increases. Also, the adversarial approach also caused the training process to be less stable and more sensitive to parameter selection.

## References
1. Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings.
2. Mescheder, L. M., Nowozin, S., & Geiger, A. (2017). Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks. CoRR, abs/1701.04722. Retrieved from http://arxiv.org/abs/1701.04722
