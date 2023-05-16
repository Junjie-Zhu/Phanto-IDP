import os
import json
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from layers import *
from utils import *


class PhantoIDP(nn.Module):

    def __init__(self, **kwargs):
        super(PhantoIDP, self).__init__()

        self.build(**kwargs)

        self.inputs = None
        self.targets = None
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        lr = kwargs.get('lr', 0.01)
        self.optimizer = optim.Adam(self.parameters(), lr, weight_decay=0)

    def build(self, **kwargs):
        # Get atom embeddings
        self.atom_init_file = os.path.join(kwargs.get('pkl_dir'), kwargs.get('atom_init'))
        with open(self.atom_init_file) as f:
            loaded_embed = json.load(f)

        embed_list = [torch.tensor(value, dtype=torch.float32) for value in loaded_embed.values()]
        self.atom_embeddings = torch.stack(embed_list, dim=0)

        self.h_init = self.atom_embeddings.shape[-1]  # Dim atom embedding init
        self.h_b = kwargs.get('h_b')  # Dim bond embedding init

        assert self.h_init is not None and self.h_b is not None

        self.h_a = kwargs.get('h_a', 64)  # Dim of the hidden atom embedding learnt
        self.n_conv = kwargs.get('n_conv', 4)  # Number of GCN layers
        self.h_g = kwargs.get('h_g', 32)  # Dim of the hidden graph embedding after pooling
        random_seed = kwargs.get('random_seed', None)  # Seed to fix the simulation

        # The model is defined below
        randomSeed(random_seed)
        self.embed = nn.Embedding.from_pretrained(self.atom_embeddings,
                                                  freeze=True)  # Load atom embeddings from the one hot atom init
        self.embedding = nn.Linear(self.h_init, self.h_a)
        self.convs = nn.ModuleList([ConvLayer(self.h_a, self.h_b, random_seed=random_seed) for _ in range(self.n_conv)])

        self.amino_to_mu = nn.Linear(self.h_a * 3, self.h_g)
        self.amino_to_var = nn.Linear(self.h_a * 3, self.h_g)
        self.amino_to_fc_activation = nn.ReLU()
        self.amino_to_fc = nn.Linear(self.h_g, 32)
        self.fc_amino_out = nn.Linear(32, 9)
        self.transformers = nn.ModuleList([IdpGANBlock(embed_dim=32,
                                                       d_model=128, nhead=8,
                                                       dim_feedforward=128,
                                                       dropout=0.1,
                                                       layer_norm_eps=1e-05,
                                                       norm_pos="post",
                                                       embed_dim_2d=None,
                                                       use_bias_2d=True,
                                                       embed_dim_1d=None,
                                                       activation="relu",
                                                       dp_attn_norm="d_model") for _ in range(self.n_conv)])

    def forward(self, inputs):

        [atom_emb_idx, nbr_emb, nbr_adj_list] = inputs

        batch_size = atom_emb_idx.size(0)

        lookup_tensor = self.embed(atom_emb_idx.type(torch.long))
        atom_emb = self.embedding(lookup_tensor)

        for idx in range(self.n_conv):
            atom_emb = self.convs[idx](atom_emb, nbr_emb, nbr_adj_list)

        # Update the embedding using the mask
        atom_emb = atom_emb.view(batch_size, -1, self.h_a * 3)

        # generate reside amino acid level embeddings
        amino_mu = self.amino_to_mu(self.amino_to_fc_activation(atom_emb))
        amino_logvar = self.amino_to_var(self.amino_to_fc_activation(atom_emb))
        amino_emb = self.reparameterize(amino_mu, amino_logvar)

        amino_emb = self.amino_to_fc(amino_emb).transpose(0, 1)

        for idx in range(self.n_conv):
            amino_emb = self.transformers[idx](amino_emb)

        amino_emb = amino_emb.transpose(0, 1)

        # [B, A, 3]
        out = self.fc_amino_out(amino_emb)

        return out.view(batch_size, -1, 3, 3), amino_mu, amino_logvar

    def sample(self, inputs):
        amino_emb = inputs
        amino_emb = self.amino_to_fc(amino_emb).transpose(0, 1)
        
        batch_size = amino_emb.shape[1]
        for idx in range(self.n_conv):
            amino_emb = self.transformers[idx](amino_emb)

        amino_emb = amino_emb.transpose(0, 1)

        # [B, A, 3]
        out = self.fc_amino_out(amino_emb)

        return out

    @staticmethod
    def reparameterize(means, logvars, temp=1.0):
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        return means + eps * std * temp

    def save(self, state, is_best, savepath, filename='checkpoint.pth.tar'):
        """Save model checkpoints"""
        torch.save(state, savepath + filename)
        if is_best:
            shutil.copyfile(savepath + filename, savepath + 'model_best.pth.tar')

    @staticmethod
    def from_3_points(
            p_neg_x_axis: torch.Tensor,
            origin: torch.Tensor,
            p_xy_plane: torch.Tensor,
            eps: float = 1e-8
    ):
        """
            Implements algorithm 21. Constructs transformations from sets of 3
            points using the Gram-Schmidt algorithm.
            Args:
                p_neg_x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        return rots, torch.stack(origin, dim=-1)

    @staticmethod
    def calc_rmsd(target, predicted):
        """Calculate optimal RMSD between two given structures"""
        target -= torch.mean(target, dim=1, keepdim=True)
        predicted -= torch.mean(predicted, dim=1, keepdim=True)

        target = torch.unbind(target, dim=0)
        predicted = torch.unbind(predicted, dim=0)

        idx, rmsd = 0, []
        for conft, confp in zip(target, predicted):
            N = conft.shape[0]
            W = torch.stack([makeW(*confp[k]) for k in range(N)])
            Q = torch.stack([makeQ(*conft[k]) for k in range(N)])
            Qt_dot_W = torch.stack([torch.mm(Q[k].T, W[k]) for k in range(N)])
            A = torch.sum(Qt_dot_W, dim=0)
            eigen = torch.linalg.eigh(A)
            r = eigen[1][:, eigen[0].argmax()]

            Wt_r = makeW(*r).T
            Q_r = makeQ(*r)
            rot: torch.Tensor = Wt_r.mm(Q_r)[:3, :3]

            conft = torch.mm(conft, rot)
            diff = conft - confp
            rmsd.append(torch.sqrt((diff * diff).sum() / conft.shape[0]))

        return rmsd

    def fit(self, outputs, targets, weight, pred=False):
        """Train the model one step for given inputs"""

        batch_size = outputs[0].shape[0]

        self.targets = targets  # (n, ca, c)
        self.outputs = torch.split(outputs[0], 1, dim=-2)

        targets_rigid = self.from_3_points(self.targets[0], self.targets[1], self.targets[2])[0]
        outputs_rigid = self.from_3_points(self.outputs[0].squeeze(),
                                           self.outputs[1].squeeze(),
                                           self.outputs[2].squeeze())[0]

        self.kl_loss = KL_loss(outputs, weight=weight[1])
        self.fape = FAPEloss()((targets_rigid, self.targets[0]), (outputs_rigid, self.outputs[0].squeeze())) + \
                    FAPEloss()((targets_rigid, self.targets[1]), (outputs_rigid, self.outputs[1].squeeze())) + \
                    FAPEloss()((targets_rigid, self.targets[2]), (outputs_rigid, self.outputs[2].squeeze()))
        self.loss = self.fape * weight[0] / 3 - self.kl_loss

        if not pred:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

