import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize_latent_space(model, dataloader):
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

    plt.scatter(tsne_representations[:, 0], tsne_representations[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.show()

def create_scatter_toy(model, device, batch_size=1000, model_type='VAE'):

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

    return fig