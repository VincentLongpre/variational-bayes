from VariNet.utils.train import train_vae_MNIST, train_avae_toy, train_avae_MNIST, train_vae_toy
from VariNet.models.vae import MnistVAE
from VariNet.models.avae import MnistAVAE
import json
import torch
from pathlib import Path

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Create examples directory
    Path("results").mkdir(parents=True, exist_ok=True)

    ##########################################################
    print('#'*50)
    print('VAE Vanilla MNIST')

    try:
        train_vae_MNIST(batch_size = 64,
                    z_dim = 32,
                    lr = 1e-3,
                    epochs = 30,
                    image_size = 32,
                    input_channels = 1,
                    results_folder = "results/vae_vanilla_MNIST",
                    binary = False,
                    decoder_features = 32,
                    encoder_features = 32)
    except:
        print('Error in VAE Vanilla MNIST')

    ##########################################################
    print('#'*50)
    print('VAE Binary MNIST')

    try:
        train_vae_MNIST(batch_size = 64,
                    z_dim = 32,
                    lr = 1e-3,
                    epochs = 30,
                    image_size = 32,
                    input_channels = 1,
                    results_folder = "results/vae_binary_MNIST",
                    binary = True,
                    decoder_features = 32,
                    encoder_features = 32)
    except:
        print('Error in VAE Binary MNIST')

    ##########################################################
    print('#'*50)
    print('AVAE Vanilla MNIST')

    print('Setup 1')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 16,
                        primal_lr = 1e-3,
                        adv_lr = 1e-4,
                        epochs = 30,
                        image_size = 16,
                        input_channels = 1,
                        results_folder = "results/avae_vanilla_mnist_1",
                        binary = False,
                        decoder_features = 16,
                        encoder_features = 16)
    except:
        print('Error in AVAE Vanilla MNIST Setup 1')

    print('Setup 2')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 32,
                        primal_lr = 1e-3,
                        adv_lr = 1e-4,
                        epochs = 30,
                        image_size = 32,
                        input_channels = 1,
                        results_folder = "results/avae_vanilla_mnist_2",
                        binary = False,
                        decoder_features = 32,
                        encoder_features = 32)
    except:
        print('Error in AVAE Vanilla MNIST Setup 2')

    print('Setup 3')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 32,
                        primal_lr = 1e-4,
                        adv_lr = 5e-4,
                        epochs = 40,
                        image_size = 32,
                        input_channels = 1,
                        results_folder = "results/avae_vanilla_mnist_3",
                        binary = False,
                        decoder_features = 32,
                        encoder_features = 32)
    except:
        print('Error in AVAE Vanilla MNIST Setup 3')

    print('Setup 4')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 64,
                        primal_lr = 1e-3,
                        adv_lr = 1e-4,
                        epochs = 40,
                        image_size = 32,
                        input_channels = 1,
                        results_folder = "results/avae_vanilla_mnist_4",
                        binary = False,
                        decoder_features = 64,
                        encoder_features = 64)
    except:
        print('Error in AVAE Vanilla MNIST Setup 4')

    ##########################################################
    print('#'*50)
    print('AVAE Binary MNIST')

    print('Setup 1')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 16,
                        primal_lr = 1e-3,
                        adv_lr = 1e-4,
                        epochs = 30,
                        image_size = 16,
                        input_channels = 1,
                        results_folder = "results/avae_binary_mnist_1",
                        binary = True,
                        decoder_features = 16,
                        encoder_features = 16)
    except:
        print('Error in AVAE Binary MNIST Setup 1')

    print('Setup 2')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 32,
                        primal_lr = 1e-3,
                        adv_lr = 1e-4,
                        epochs = 30,
                        image_size = 32,
                        input_channels = 1,
                        results_folder = "results/avae_binary_mnist_2",
                        binary = True,
                        decoder_features = 32,
                        encoder_features = 32)
    except:
        print('Error in AVAE Binary MNIST Setup 2')

    print('Setup 3')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 32,
                        primal_lr = 1e-4,
                        adv_lr = 5e-4,
                        epochs = 40,
                        image_size = 32,
                        input_channels = 1,
                        results_folder = "results/avae_binary_mnist_3",
                        binary = True,
                        decoder_features = 32,
                        encoder_features = 32)
    except:
        print('Error in AVAE Binary MNIST Setup 3')

    print('Setup 4')
    try:
        train_avae_MNIST(batch_size = 64,
                        z_dim = 64,
                        primal_lr = 1e-3,
                        adv_lr = 1e-4,
                        epochs = 40,
                        image_size = 32,
                        input_channels = 1,
                        results_folder = "results/avae_binary_mnist_4",
                        binary = True,
                        decoder_features = 64,
                        encoder_features = 64)
    except:
        print('Error in AVAE Binary MNIST Setup 4')
