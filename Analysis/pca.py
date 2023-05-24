import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

dat_path = sys.argv[1]


def coords_to_dist(coord):
    # Input: B x n x 3.
    n = coord.size(1)
    G = torch.bmm(coord, coord.transpose(1, 2))
    Gt = torch.diagonal(G, dim1=-2, dim2=-1)[:, None, :]
    Gt = Gt.repeat(1, n, 1)
    dm = Gt + Gt.transpose(1, 2) - 2 * G
    dm = torch.sqrt(dm)[:, None, :, :]
    return dm.squeeze()


coords = []
for dat in os.listdir(dat_path):
    if dat.endswith('.dat') and dat.startswith('predicted'):
        coords.append(np.loadtxt(os.path.join(dat_path, dat)))
    if len(coords) >= 60:
        break
coords = torch.Tensor(coords)
coords = coords.view(-1, 20, coords.shape[-2], coords.shape[-1])

dists = torch.zeros((coords.shape[0], coords.shape[1], coords.shape[2], coords.shape[2]))
for i in range(coords.shape[0]):
    dists[i] = coords_to_dist(coords[i])

dists = dists.view(dists.shape[0], dists.shape[1], -1)
dists_to_pca = dists.view(-1, dists.shape[-1]).numpy()

pca = PCA(n_components=2)
to_plot = pca.fit_transform(dists_to_pca)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(to_plot[:0], to_plot[:1])
plt.show()
