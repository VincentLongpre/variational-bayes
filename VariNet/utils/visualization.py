import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F

def latent_dim_reduction(model, dataloader, device):
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
            x = imgs.to(device)

            posterior = model.encode(x)
            z = posterior.sample()
            latent_representations_list.append(z.cpu())
            labels_list.extend(labels.cpu())

    latent_representations = torch.cat(latent_representations_list, dim=0)
    labels = torch.tensor(labels_list)

    tsne = TSNE(perplexity=20, verbose=1)
    tsne_representations = tsne.fit_transform(latent_representations)

    pca = PCA(n_components=2)
    pca_representations= pca.fit_transform(latent_representations)

    return tsne_representations, pca_representations, labels


def scatter(data, labels, title, save_path):
    # Scatter PCA - VAE
    fig = plt.figure(figsize=(7, 6))

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', alpha=1, s = 1)
    plt.colorbar()
    plt.title(title)

    #Remove ticks
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def full_scatter(representations, labels, titles, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(4):
        ind_i, ind_j = i // 2, i % 2
        axs[ind_i, ind_j].scatter(representations[i][:, 0], representations[i][:, 1], c=labels[i], cmap='tab10', alpha=1, s = 1)
        axs[ind_i, ind_j].set_title(titles[i])

    #Remove ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(axs[0, 0].collections[0], cax=cbar_ax)

    plt.savefig(save_path, dpi=100, bbox_inches='tight')


def save_images(images, save_path):
    a = int(np.sqrt(images.shape[0]))
    b = images.shape[0] // a
    images = images.detach().to('cpu').squeeze()

    fig, ax = plt.subplots(a, b, figsize=(b-0.1, a))
    for i in range(a):
        for j in range(b):
            ax[i, j].imshow(images[i*b+j], cmap='gray')
            ax[i, j].axis('off')

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(save_path, dpi=200)


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


def create_reconstructions(vae, avae, test_dataloader, device, save_path, n=5):
    vae.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        imgs, _ = batch
        x = imgs.to(device)

        posterior = vae.encode(x)
        z = posterior.mode()
        x_hat_vae = vae.decode(z).mode()

        x_hat_vae = x_hat_vae.cpu()

    avae.eval()
    with torch.no_grad():
        x = imgs.to(device)

        z, _, _ = avae.encode(x)
        x_hat_avae = avae.decode(z).mode()

        x = x.cpu()
        x_hat_avae = x_hat_avae.cpu()
    
    fig, axs = plt.subplots(6, n//2, figsize=(n//2, 6))

    for i in range(n):
        if i%2 == 0:
            axs[2, i//2].imshow(x[i].squeeze(), cmap='gray')
            axs[0, i//2].imshow(x_hat_vae[i].squeeze(), cmap='gray')
            axs[4, i//2].imshow(x_hat_avae[i].squeeze(), cmap='gray')
        else:
            axs[3, i//2].imshow(x[i].squeeze(), cmap='gray')
            axs[1, i//2].imshow(x_hat_vae[i].squeeze(), cmap='gray')
            axs[5, i//2].imshow(x_hat_avae[i].squeeze(), cmap='gray')

    # Remove ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Remve horizontal space between axes
    fig.subplots_adjust(hspace=0, wspace=0)

    axs[0, 0].set_ylabel('VAE')
    axs[2, 0].set_ylabel('Original')
    axs[4, 0].set_ylabel('AVB')

    plt.suptitle("Reconstructions - VAE and AVB")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


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