from ..models.vae import ToyVAE, MnistVAE
from ..models.avae import MnistAVAE, ToyAVAE
from ..utils.datasets import toy_dataloader, mnist_dataloaders, binary_mnist_dataloaders
from ..utils.visualization import create_scatter_toy

import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torchvision.utils import save_image

from tqdm.auto import tqdm
from pathlib import Path
import json

data_root = './data'

def train_vae_MNIST(batch_size = 64,
                z_dim = 32,
                lr = 1e-3,
                epochs = 10,
                image_size = 32,
                input_channels = 1,
                results_folder = "./results",
                binary = False,
                decoder_features = 32,
                encoder_features = 32,
                download = True):

    # Training setup
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create image directory inside results folder
    images_folder = results_folder / 'images'
    images_folder.mkdir(exist_ok=True)

    # Create model, and optimizer
    vae = MnistVAE(in_channels=input_channels,
            input_size=image_size,
            z_dim=z_dim,
            decoder_features=decoder_features,
            encoder_features=encoder_features,
            device=device)
    vae.to(device)
    optimizer = Adam(vae.parameters(), lr=lr)

    # Get MNIST dataloaders (binary or not)
    if binary:
        train_dataloader, _ = binary_mnist_dataloaders(data_root, batch_size=batch_size, image_size=image_size, download=download)
    else:
        train_dataloader, _ = mnist_dataloaders(data_root, batch_size=batch_size, image_size=image_size, download=download)

    # Training loop
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            vae.train()
            nb_batches = len(tepoch)
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()

                imgs, _ = batch
                batch_size = imgs.shape[0]
                x = imgs.to(device)

                recon, nll, kl = vae(x)
                loss = (nll + kl).mean()
                epoch_loss += loss

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        epoch_loss /= nb_batches
        print(f'Epoch {epoch} Mean Loss: {epoch_loss}')
        train_losses.append(epoch_loss)

        # Save image samples to results folder (from params)
        samples = vae.sample(batch_size=64)
        save_image(samples, images_folder / f"epoch_{epoch}.png", nrow=8)

    # Train losses to list of floats
    train_losses = [float(loss) for loss in train_losses]

    # Create dictionary with training parameters
    params = {'model_type': 'VAE',
              'batch_size': batch_size,
              'z_dim': z_dim,
              'lr': lr,
              'epochs': epochs,
              'image_size': image_size,
              'input_channels': input_channels,
              'binary': binary,
              'decoder_features': decoder_features,
              'encoder_features': encoder_features,
              'train_losses': train_losses,
              'dataset': 'MNIST'}

    # Save training parameters to results folder
    with open(results_folder / 'params.json', 'w') as f:
        json.dump(params, f)

    # Save model state dict
    torch.save(vae.state_dict(), results_folder / 'model.pth')

    # Clean gpu
    del vae
    torch.cuda.empty_cache()

def train_vae_toy(batch_size = 512,
                lr = 1e-4,
                results_folder = "./results",
                nb_samples = 10000):

    # Training setup
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create image directory inside results folder
    images_folder = results_folder / 'images'
    images_folder.mkdir(exist_ok=True)

    # Create model, and optimizer
    vae = ToyVAE(input_size=4,
                z_dim=2,
                device=device)
    vae.to(device)
    vae_optimizer = Adam(vae.parameters(), lr=lr)

    # Get Toy dataloader
    train_dataloader = toy_dataloader(batch_size=batch_size)

    # Training loop
    vae.train()
    for i in tqdm(range(nb_samples)):
        batch = next(train_dataloader)
        vae_optimizer.zero_grad()

        imgs = batch
        batch_size = imgs.shape[0]
        x = imgs.to(device)

        recon, nll, kl = vae(x)
        loss = (nll+0.1*kl).mean()

        loss.backward()
        vae_optimizer.step()

    # Create dictionary with training parameters
    params = {'model_type': 'VAE',
              'batch_size': batch_size,
              'lr': lr,
              'nb_samples': nb_samples,
              'dataset': 'toy'}

    # Save training parameters to results folder
    with open(results_folder / 'params.json', 'w') as f:
        json.dump(params, f)

    # Save model state dict
    torch.save(vae.state_dict(), results_folder / 'model.pth')

    # Create and save scatter plot
    fig = create_scatter_toy(vae, device, model_type='VAE')
    fig.savefig(images_folder / 'scatter.png')

    # Clean gpu
    del vae
    torch.cuda.empty_cache()

def train_avae_MNIST(batch_size = 64,
                z_dim = 32,
                primal_lr = 1e-3,
                adv_lr = 1e-4,
                epochs = 10,
                image_size = 32,
                input_channels = 1,
                results_folder = "./results",
                binary = False,
                decoder_features = 32,
                encoder_features = 32,
                download = True):

    # Training setup
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create image directory inside results folder
    images_folder = results_folder / 'images'
    images_folder.mkdir(exist_ok=True)

    # Create model, and optimizer
    avae = MnistAVAE(in_channels=input_channels,
                    input_size=image_size,
                    z_dim=z_dim,
                    decoder_features=decoder_features,
                    encoder_features=encoder_features,
                    device=device)
    avae.to(device)
    avae_primal_optimizer = Adam([param for name, param in avae.named_parameters() if 'adversary' not in name], lr=primal_lr)
    avae_adv_optimizer = Adam(avae.adversary.parameters(), lr=adv_lr)

    # Get MNIST dataloaders (binary or not)
    if binary:
        train_dataloader, _ = binary_mnist_dataloaders(data_root, batch_size=batch_size, image_size=image_size, download=download)
    else:
        train_dataloader, _ = mnist_dataloaders(data_root, batch_size=batch_size, image_size=image_size, download=download)

    # Function to zero out gradients
    def zero_grad():
        avae_primal_optimizer.zero_grad()
        avae_adv_optimizer.zero_grad()

    # Training loop
    train_losses = []
    recon_error = []
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            avae.train()
            nb_batches = len(tepoch)
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                zero_grad()

                imgs, _ = batch
                batch_size = imgs.shape[0]
                x = imgs.to(device)

                for _ in range(2):
                    recon, nll, kl, adv = avae(x)
                    dual_loss = adv.mean()

                    dual_loss.backward()
                    avae_adv_optimizer.step()
                    zero_grad()

                # Backward pass for primal loss
                recon, nll, kl, adv = avae(x)
                primal_loss = (nll+kl).mean()
                epoch_loss += primal_loss
                epoch_recon_error += mse_loss(recon, x)

                primal_loss.backward()
                avae_primal_optimizer.step()

                tepoch.set_postfix(loss=primal_loss.item())

        epoch_loss /= nb_batches
        epoch_recon_error = epoch_recon_error / nb_batches
        print(f'Epoch {epoch} Mean Loss: {epoch_loss} Mean Recon Error: {epoch_recon_error}')
        train_losses.append(epoch_loss)
        recon_error.append(epoch_recon_error)

        # Save image samples to results folder (from params)
        samples = avae.sample(batch_size=64)
        save_image(samples, images_folder / f"epoch_{epoch}.png", nrow=8)

    # Train losses to list of floats
    train_losses = [float(loss) for loss in train_losses]

    # Create dictionary with training parameters
    params = {'model_type': 'AVAE',
              'batch_size': batch_size,
              'z_dim': z_dim,
              'primal_lr': primal_lr,
              'adv_lr': adv_lr,
              'epochs': epochs,
              'image_size': image_size,
              'input_channels': input_channels,
              'binary': binary,
              'decoder_features': decoder_features,
              'encoder_features': encoder_features,
              'train_losses': train_losses}

    # Save training parameters to results folder
    with open(results_folder / 'params.json', 'w') as f:
        json.dump(params, f)

    # Save model state dict
    torch.save(avae.state_dict(), results_folder / 'model.pth')

    # Clean gpu
    del avae
    torch.cuda.empty_cache()

def train_avae_toy(batch_size = 512,
                primal_lr = 1e-4,
                adv_lr = 1e-4,
                results_folder = "./results",
                nb_samples = 10000):

    # Training setup
    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create image directory inside results folder
    images_folder = results_folder / 'images'
    images_folder.mkdir(parents=True, exist_ok=True)

    # Create model, and optimizer
    avae = ToyAVAE(input_size=4,
                z_dim=2,
                device=device
                )
    avae.to(device)
    avae_primal_optimizer = Adam([param for name, param in avae.named_parameters() if 'adversary' not in name], lr=primal_lr)
    avae_adv_optimizer = Adam(avae.adversary.parameters(), lr=adv_lr)

    # Function to zero out gradients
    def zero_grad():
        avae_primal_optimizer.zero_grad()
        avae_adv_optimizer.zero_grad()

    # Get Toy dataloader
    train_dataloader = toy_dataloader(batch_size=batch_size)

    # Training loop
    avae.train()
    for i in tqdm(range(nb_samples)):
        batch = next(train_dataloader)
        zero_grad()

        imgs = batch
        batch_size = imgs.shape[0]
        x = imgs.to(device)

        for _ in range(2):
            recon, nll, kl, adv = avae(x)
            dual_loss = adv.mean()

            dual_loss.backward()
            avae_adv_optimizer.step()
            zero_grad()

        # Backward pass for primal loss
        recon, nll, kl, adv = avae(x)
        primal_loss = (nll+0.1*kl).mean()
        primal_loss.backward()
        avae_primal_optimizer.step()

    # Create dictionary with training parameters
    params = {'model_type': 'VAE',
              'batch_size': batch_size,
              'primal_lr': primal_lr,
              'adv_lr': adv_lr,
              'nb_samples': nb_samples,
              'dataset': 'toy'}

    # Save training parameters to results folder
    with open(results_folder / 'params.json', 'w') as f:
        json.dump(params, f)

    # Save model state dict
    torch.save(avae.state_dict(), results_folder / 'model.pth')

    # Create and save scatter plot
    fig = create_scatter_toy(avae, device, model_type='AVAE')
    fig.savefig(images_folder / 'scatter.png')

    # Clean gpu
    del avae
    torch.cuda.empty_cache()
