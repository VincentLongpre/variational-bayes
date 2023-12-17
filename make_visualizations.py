from VariNet.models.vae import MnistVAE
from VariNet.models.avae import MnistAVAE
from VariNet.utils.datasets import binary_mnist_dataloaders
from VariNet.utils.visualization import visualize_latent_space, visualize_images
import json
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "./data"
result_root = "./VariNet/plots"
model_root = "./VariNet/trained_models/"

with open(model_root + 'avae_best/params.json', 'r') as json_file:
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
    avae.load_state_dict(torch.load(model_root + 'avae_best/model.pth'))
    avae.to(device)

    vae = MnistVAE(in_channels=input_channels,
            input_size=image_size,
            z_dim=z_dim,
            decoder_features=decoder_features,
            encoder_features=encoder_features,
            device=device)
    vae.load_state_dict(torch.load(model_root + 'vae_comparison/model.pth'))
    vae.to(device)

    # Visualize latent space
    visualize_latent_space(vae, test_dataloader, device, result_root+"vae_latent.png")
    visualize_latent_space(avae, test_dataloader, device, result_root+"avb_latent.png")

    # Visualize samples
    vae_samples = vae.sample(batch_size)
    avae_samples = avae.sample(batch_size)
    
    visualize_images(vae_samples, result_root+"vae_samples.png")
    visualize_images(avae_samples, result_root+"avb_samples.png")

    # Visualize interpolations

