from time import time

import matplotlib.pyplot as plt

# import sys
# sys.path.append("..")
import numpy as np
import torch
from sklearn.manifold import TSNE

from lib.dataset import makeDataLoader
from lib.loss import makeLoss
from lib.models import makeModel

device = torch.device("cuda:0")
# data
n_samples, classes, testloader = makeDataLoader(
    train=False, bw=[32, 16, 8], dataset_name="modelnet10"
)
# loss
criterion = makeLoss("margin_loss", nclass=len(classes))
# model
_, model = makeModel(
    "msvc",
    "../logs/msvc_modelnet10_marginloss_norotate",
    nclass=len(classes),
    device=device,
    is_distributed=True,
    use_residual_block=True,
    load=True,
)


def get_embeddings(model, testloader):
    model.eval()
    embeddings = []
    labels = []
    for batch_idx, (data, target) in enumerate(testloader):
        if isinstance(data, list):
            inputs = [i.to(device) for i in data]
            data = inputs
        else:
            data = data.to(device)
        prediction = model(data)
        embeddings.append(prediction.cpu().detach().numpy())
        labels.append(target.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_samples, n_features = embeddings.shape
    return embeddings, labels, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(
            data[i, 0],
            data[i, 1],
            str(label[i]),
            color=plt.cm.Set1(label[i] / 10.0),
            fontdict={"weight": "bold", "size": 9},
        )
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.figure(dpi=300, figsize=(24, 16))
    return fig


def main(func=None):
    if not func:
        return
    data, label, n_samples, n_features = func(model, testloader)
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label, "t-SNE embedding of ModelNet10" % (time() - t0))
    plt.show(fig)


main(get_embeddings)
