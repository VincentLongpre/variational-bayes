from VariNet.models.vae import MnistVAE
from VariNet.models.avae import MnistAVAE
from VariNet.utils.datasets import binary_mnist_dataloaders
from VariNet.utils.visualization import *
import json
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./data"
result_root = "./figures"
model_root = "./trained_models"

with open(model_root + '/AVB/params.json', 'r') as json_file:
    params = json.load(json_file)

locals().update(params)
download = True


if __name__ == "__main__":
    train_dataloader, test_dataloader = binary_mnist_dataloaders(data_root, batch_size=batch_size, image_size=image_size, download=download)

    # Load models
    avae = MnistAVAE(in_channels=input_channels,
                    input_size=image_size,
                    z_dim=z_dim,
                    decoder_features=decoder_features,
                    encoder_features=encoder_features,
                    device=device)
    avae.load_state_dict(torch.load(model_root + '/AVB/model.pth'))
    avae.to(device)

    vae = MnistVAE(in_channels=input_channels,
            input_size=image_size,
            z_dim=z_dim,
            decoder_features=decoder_features,
            encoder_features=encoder_features,
            device=device)
    vae.load_state_dict(torch.load(model_root + '/VAE/model.pth'))
    vae.to(device)

    # Visualize latent space
    vae_tsne_latent, vae_pca_latent, vae_labels = latent_dim_reduction(vae, test_dataloader, device)
    avae_tsne_latent, avae_pca_latent, avae_labels = latent_dim_reduction(avae, test_dataloader, device)

    scatter(vae_pca_latent, vae_labels, 'PCA - VAE', result_root+"pca_vae.png")
    scatter(avae_pca_latent, avae_labels, 'PCA - AVB', result_root+"pca_avb.png")
    scatter(vae_tsne_latent, vae_labels, 't-SNE - VAE', result_root+"tsne_vae.png")
    scatter(avae_tsne_latent, avae_labels, 't-SNE - AVB', result_root+"tsne_avb.png")

    latent_representations = [vae_pca_latent, avae_pca_latent, vae_tsne_latent, avae_tsne_latent]
    latent_labels = [vae_labels, avae_labels, vae_labels, avae_labels]
    latent_titles = ['PCA - VAE', 'PCA - AVB', 't-SNE - VAE', 't-SNE - AVB']
    full_scatter(latent_representations, latent_labels, latent_titles, result_root+"latent_space.png")

    # Visualize samples
    vae_samples = vae.sample(batch_size)
    avae_samples = avae.sample(batch_size)

    save_images(vae_samples, result_root+"vae_samples.png")
    save_images(avae_samples, result_root+"avb_samples.png")

    # Visualize interpolations
    interpolate(vae, z_dim, device, result_root+"vae_interpolate.png")
    interpolate(avae, z_dim, device, result_root+"avb_interpolate.png")

    # Create reconstructions
    create_reconstructions(vae, avae, test_dataloader, device, result_root+"reconstructions.png")

