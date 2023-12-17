import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize_latent_space(model, dataloader, device, save_path):
    """
        Represent the latent representation of each class for the given model.

        Parameters:
            model: The trained torch model.
            dataloader: The dataloader for the testing data.
    """
    model.eval()
    latent_representations_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch
            batch_size = imgs.shape[0]
            x = imgs.to(device)

            posterior = model.encode(x)
            z = posterior.sample()
            latent_representations_list.append(z.cpu())
            labels_list.extend(labels.cpu())

    latent_representations = torch.cat(latent_representations_list, dim=0)
    labels = torch.tensor(labels_list)
    tsne = TSNE(perplexity=20, verbose=1)
    tsne_representations = tsne.fit_transform(latent_representations)

    plt.scatter(tsne_representations[:, 0], tsne_representations[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
    plt.colorbar()
    plt.savefig(save_path)

def save_images(images, save_path):
    a = int(np.sqrt(images.shape[0]))
    b = images.shape[0] // a
    images = images.detach().to('cpu').squeeze()

    fig, ax = plt.subplots(a, b, figsize=(b-0.1, a))
    for i in range(a):
        for j in range(b):
            ax[i, j].imshow(images[i*b+j], cmap='gray')
            ax[i, j].axis('off')

    plt.figure(dpi=100)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(save_path)

def interpolate(model, z_dim, device, save_path):
    z_1 = torch.randn(1, z_dim).to(device)
    z_2 = torch.randn(1, z_dim).to(device)
    z_3 = torch.randn(1, z_dim).to(device)
    z_4 = torch.randn(1, z_dim).to(device)

    lengths = torch.linspace(0., 1., 10).unsqueeze(1).to(z_1.device)

    z_h_top = z_2*lengths + z_1*(1-lengths)
    z_h_down = z_4*lengths + z_3*(1-lengths)

    for i in range(10):
        z_ver = z_h_top[i]*lengths + z_h_down[i]*(1-lengths)
        output = model.decode(z_ver).mode()
        if i == 0:
            grid = output
        else:
            grid = torch.cat((grid, output), 0)

    save_images(grid, save_path)

def create_scatter_toy(model, device, save_path, batch_size=1000, model_type='VAE'):

    # Create figure to return
    fig = plt.figure(figsize=(8,8))

    for label in range(4):
        images = F.one_hot(torch.tensor([label]), 4).float().repeat(batch_size, 1)

        if model_type == 'VAE':
            posterior = model.encode(images.to(device))
            latent_z = posterior.sample()

        elif model_type == 'AVAE':
            latent_z = model.encode(images.to(device))

        latent_z = latent_z.detach().to('cpu')
        plt.scatter(latent_z[:, 0], latent_z[:, 1],  edgecolor='none', alpha=0.5, label=f"Class {label}")

    plt.xlim(-3, 3); plt.ylim(-3.5, 3.5)
    plt.legend()
    plt.savefig(save_path)