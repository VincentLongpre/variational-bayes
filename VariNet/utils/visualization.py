import torch
from sklearn.manifold import TSNE

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