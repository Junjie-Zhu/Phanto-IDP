import torch
import torch.nn as nn

from utils import *


class ConvLayer(nn.Module):

    def __init__(self, h_a, h_b, random_seed=None):
        randomSeed(random_seed)
        super(ConvLayer, self).__init__()
        self.h_a = h_a
        self.h_b = h_b
        self.fc_full = nn.Linear(2 * self.h_a + self.h_b, 2 * self.h_a)
        self.sigmoid = nn.Sigmoid()
        self.activation_hidden = nn.ReLU()
        self.bn_hidden = nn.BatchNorm1d(2 * self.h_a)
        self.bn_output = nn.BatchNorm1d(self.h_a)
        self.activation_output = nn.ReLU()

    def forward(self, atom_emb, nbr_emb, nbr_adj_list):
        N, M = nbr_adj_list.shape[1:]
        B = atom_emb.shape[0]

        atom_nbr_emb = atom_emb[torch.arange(B).unsqueeze(-1), nbr_adj_list.view(B, -1)].view(B, N, M, self.h_a)

        total_nbr_emb = torch.cat([atom_emb.unsqueeze(2).expand(B, N, M, self.h_a), atom_nbr_emb, nbr_emb], dim=-1)
        total_gated_emb = self.fc_full(total_nbr_emb)
        total_gated_emb = self.bn_hidden(total_gated_emb.view(-1, self.h_a * 2)).view(B, N, M, self.h_a * 2)
        nbr_filter, nbr_core = total_gated_emb.chunk(2, dim=3)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.activation_hidden(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=2)
        nbr_sumed = self.bn_output(nbr_sumed.view(-1, self.h_a)).view(B, N, self.h_a)
        out = self.activation_output(atom_emb + nbr_sumed)

        return out


class IdpGANBlock(nn.Module):

    def __init__(self, embed_dim, d_model=192, nhead=12,
                 dim_feedforward=128,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 norm_pos="post",
                 embed_dim_2d=None,
                 use_bias_2d=True,
                 embed_dim_1d=None,
                 activation="relu",
                 dp_attn_norm="d_model"):

        super(IdpGANBlock, self).__init__()

        self.use_norm = layer_norm_eps is not None
        self.norm_pos = norm_pos
        if not norm_pos in ("post", "pre"):
            raise KeyError(norm_pos)
        self.use_embed_2d = embed_dim_2d is not None
        self.use_embed_1d = embed_dim_1d is not None
        self.use_dropout = dropout is not None
        _dropout = dropout if dropout is not None else 0.0

        # Transformer layer.
        self.idp_attn = IdpGANLayer(in_dim=embed_dim,
                                    d_model=d_model,
                                    nhead=nhead,
                                    dp_attn_norm=dp_attn_norm,
                                    in_dim_2d=embed_dim_2d,
                                    use_bias_2d=use_bias_2d)

        contact_embed_1d_dim = embed_dim_1d
        if embed_dim_1d == None:
            contact_embed_1d_dim = 0
        updater_in_dim = embed_dim + contact_embed_1d_dim

        # Updater module (implementation of Feedforward model of the original
        # transformer).
        self.linear1 = nn.Linear(updater_in_dim, dim_feedforward)
        self.dropout = nn.Dropout(_dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        if self.use_norm:
            self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            if self.norm_pos == "post":
                self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            elif self.norm_pos == "pre":
                self.pre_linear = nn.Linear(updater_in_dim, updater_in_dim)
                self.norm2 = nn.LayerNorm(updater_in_dim, eps=layer_norm_eps)
            else:
                raise KeyError(self.norm_pos)
        self.dropout1 = nn.Dropout(_dropout)
        self.dropout2 = nn.Dropout(_dropout)
        self.activation = get_activation(activation)

        self.update_module = [self.linear1, self.activation]
        if self.use_dropout:
            self.update_module.append(self.dropout)
        self.update_module.append(self.linear2)
        self.update_module = nn.Sequential(*self.update_module)

    def _check_embedding(self, use_embed, embed, var_name, embed_name):
        if use_embed:
            if embed is None:
                raise ValueError("'%s' can not be None when using %s embeddings." % (
                    var_name, embed_name))
        else:
            if embed is not None:
                raise ValueError("'%s' must be None when using %s embeddings." % (
                    var_name, embed_name))

    def forward(self, s, x=None, p=None):
        # Check the input.
        self._check_embedding(self.use_embed_2d, p, "p", "2d")
        self._check_embedding(self.use_embed_1d, x, "x", "1d")

        # Actually run the transformer block.
        if self.use_norm and self.norm_pos == "pre":
            s = self.norm1(s)
        s2 = self.idp_attn(s, s, s, p=p)[0]
        if self.use_dropout:
            s = s + self.dropout1(s2)
        else:
            s = s + s2
        if self.use_norm and self.norm_pos == "post":
            s = self.norm1(s)

        # Use amino acid conditional information.
        if self.use_embed_1d:
            um_in = torch.cat([s, x], axis=-1)
        else:
            um_in = s

        # Use the updater module.
        if self.use_norm and self.norm_pos == "pre":
            um_in = self.norm2(self.pre_linear(um_in))
        s2 = self.update_module(um_in)
        if self.use_dropout:
            s = s + self.dropout2(s2)
        else:
            s = s + s2
        if self.use_norm and self.norm_pos == "post":
            s = self.norm2(s)
        return s


class IdpGANLayer(nn.Module):

    def __init__(self, in_dim,
                 d_model, nhead,
                 dp_attn_norm="d_model",
                 in_dim_2d=None,
                 use_bias_2d=True):
        super(IdpGANLayer, self).__init__()
        """d_model = c*n_head"""

        head_dim = d_model // nhead
        assert head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = int(nhead)
        self.head_dim = head_dim
        self.in_dim_2d = in_dim_2d

        if dp_attn_norm not in ("d_model", "head_dim"):
            raise KeyError("Unkown 'dp_attn_norm': %s" % dp_attn_norm)
        self.dp_attn_norm = dp_attn_norm

        # Linear layers for q, k, v for dot product affinities.
        self.q_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.k_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.v_linear = nn.Linear(in_dim, self.d_model, bias=False)

        # Output layer.
        out_linear_in = self.d_model
        self.out_linear = nn.Linear(out_linear_in, in_dim)

        # Branch for 2d representation.
        self.mlp_2d = nn.Sequential(  # nn.Linear(in_dim_2d, in_dim_2d),
            # nn.ReLU(),
            # nn.Linear(in_dim_2d, in_dim_2d),
            # nn.ReLU(),
            nn.Linear(in_dim, self.nhead, bias=use_bias_2d))

    verbose = False

    def forward(self, s, _k, _v, p=None):

        # ----------------------
        # Prepare the  input. -
        # ----------------------

        # Receives a (L, N, I) tensor.
        # L: sequence length,
        # N: batch size,
        # I: input embedding dimension.
        seq_l, b_size, _e_size = s.shape
        if self.dp_attn_norm == "d_model":
            w_t = 1 / np.sqrt(self.d_model)
        elif self.dp_attn_norm == "head_dim":
            w_t = 1 / np.sqrt(self.head_dim)
        else:
            raise KeyError(self.dp_attn_norm)

        # ----------------------------------------------
        # Compute q, k, v for dot product affinities. -
        # ----------------------------------------------

        # Compute q, k, v vectors. Will reshape to (L, N, D*H).
        # D: number of dimensions per head,
        # H: number of head,
        # E = D*H: embedding dimension.
        q = self.q_linear(s)
        k = self.k_linear(s)
        v = self.v_linear(s)

        # Actually compute dot prodcut affinities.
        # Reshape first to (N*H, L, D).
        q = q.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        q = q * w_t
        k = k.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)

        # Then perform matrix multiplication between two batches of matrices.
        # (N*H, L, D) x (N*H, D, L) -> (N*H, L, L)
        dp_aff = torch.bmm(q, k.transpose(-2, -1))

        # --------------------------------
        # Compute the attention values. -
        # --------------------------------

        tot_aff = dp_aff
        attn = nn.functional.softmax(tot_aff, dim=-1)
        # if dropout_p > 0.0:
        #     attn = dropout(attn, p=dropout_p)

        # -----------------
        # Update values. -
        # -----------------

        # Update values obtained in the dot product affinity branch.
        s_new = torch.bmm(attn, v)
        # Reshape the output, that has a shape of (N*H, L, D) back to (L, N, D*H).
        s_new = s_new.transpose(0, 1).contiguous().view(seq_l, b_size, self.d_model)

        # Compute the ouput.
        output = s_new
        output = self.out_linear(output)
        return output