from VariNet.utils.train import train_vae_MNIST, train_avae_toy, train_avae_MNIST, train_vae_toy
from VariNet.models.vae import MnistVAE
from VariNet.models.avae import MnistAVAE
import json
import random
import torch
from pathlib import Path

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # Create examples directory
    Path("results").mkdir(parents=True, exist_ok=True)

    ##########################################################
    print('#'*50)
    print('Random Grid Search')

    # Define ranges for hyperparameters
    batch_sizes = [32, 64, 128]
    z_dims = [16, 32, 64, 100]
    lr_values = [5e-3, 1e-3, 5e-4, 1e-4]
    adv_lr_values = [5e-3, 1e-3, 5e-4, 1e-4]
    epoch_values = [2, 5, 10, 20, 30]
    image_sizes = [28, 32]
    channels = 1
    binary_values = [0, 1]
    decoder_features_list = [16, 32, 64]
    encoder_features_list = [16, 32, 64]

    for i in range(50):  # Number of setups
        print(f'Setup {i + 1}')
        try:
            # Randomly sample hyperparameters for each setup
            batch_size = random.choice(batch_sizes)
            z_dim = random.choice(z_dims)
            lr = random.choice(lr_values)
            adv_lr = random.choice(adv_lr_values)
            epochs = random.choice(epoch_values)
            image_size = random.choice(image_sizes)
            binary = random.choice(binary_values)
            decoder_features = random.choice(decoder_features_list)
            encoder_features = random.choice(encoder_features_list)

            if i % 2 == 0:  # VAE setups
                train_vae_MNIST(batch_size=batch_size,
                                z_dim=z_dim,
                                lr=lr,
                                epochs=epochs,
                                image_size=image_size,
                                input_channels=channels,
                                results_folder=f"results/vae_setup_{i + 1 if not i else i}",
                                binary=binary,
                                decoder_features=decoder_features,
                                encoder_features=encoder_features)
            else:  # AVAE setups
                train_avae_MNIST(batch_size=batch_size,
                                 z_dim=z_dim,
                                 primal_lr=lr,
                                 adv_lr=adv_lr,
                                 epochs=epochs,
                                 image_size=image_size,
                                 input_channels=channels,
                                 results_folder=f"results/avae_setup_{i}",
                                 binary=binary,
                                 decoder_features=decoder_features,
                                 encoder_features=encoder_features)

        except Exception as e:
            print(f'Error in Setup {i + 1}: {e}')
