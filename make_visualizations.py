from VariNet.utils.train import train_vae_MNIST, train_avae_MNIST
from VariNet.models.vae import MnistVAE
from VariNet.models.avae import MnistAVAE
from VariNet.utils.datasets import binary_mnist_dataloaders
import json
import random
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./data"
result_root = "./VariNet/plots"
model_root = "./VariNet/trained_models"

params = pd.read_json(model_root + 'avae_best/params.json')
locals().update(params)
download = True


if __name__ == "__main__":
    train_dataloader, test_dataloader = binary_mnist_dataloaders(data_root, batch_size=batch_size, image_size=image_size, download=download)

    avae = MnistAVAE(in_channels=input_channels,
                    input_size=image_size,
                    z_dim=z_dim,
                    decoder_features=decoder_features,
                    encoder_features=encoder_features,
                    device=device)
    avae.load_state_dict(torch.load(model_root + 'avae_best/model.pth'))
    avae.to(device)

    