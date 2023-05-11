import sys
import torch

from einops import rearrange
from torch import nn

import config as cfg


class Logger(object):
    """Writes both to file and terminal"""

    def __init__(self, savepath, mode='a'):
        self.terminal = sys.stdout
        self.log = open(savepath + 'logfile.log', mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor).type(cfg.FloatTensor)
        self.std = torch.std(tensor).type(cfg.FloatTensor)

    def norm(self, tensor):
        if self.mean != self.mean or self.std != self.std:
            return tensor
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        if self.mean != self.mean or self.std != self.std:
            return normed_tensor
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter(object):
    """
Computes and stores the average and current value. Accomodates both numbers and tensors.
If the input to be monitored is a tensor, also need the dimensions/shape of the tensor.
Also, for tensors, it keeps a column wise count for average, sum etc.
"""

    def __init__(self, is_tensor=False, dimensions=None):
        if is_tensor and dimensions is None:
            print('Bad definition of AverageMeter!')
            sys.exit(1)
        self.is_tensor = is_tensor
        self.dimensions = dimensions
        self.reset()

    def reset(self):
        self.count = 0
        if self.is_tensor:
            self.val = torch.zeros(self.dimensions, device=cfg.device)
            self.avg = torch.zeros(self.dimensions, device=cfg.device)
            self.sum = torch.zeros(self.dimensions, device=cfg.device)
        else:
            self.val = 0
            self.avg = 0
            self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FAPEloss(nn.Module):
    """Frame aligned point error loss
    Args:
        Z (int, optional): [description]. Defaults to 10.
        clamp (int, optional): [description]. Defaults to 10.
        epsion (float, optional): [description]. Defaults to -1e4.
    """
    def __init__(self, Z=10.0, clamp=10.0, epsion=-1e8):

        super().__init__()
        self.z = Z
        self.epsion = epsion
        self.clamp = clamp

    def forward(self, predict_T, transformation, pdb_mask=None, padding_mask=None, device='cuda'):
        """
        Args:
            predict_T (`tensor`, `tensor`): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
            transformation (`tensor`, `tensor`): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
            pdb_mask (`tensor`, optional): pdb mask. size: [batch, N_seq, N_seq]. Defaults to None.
            padding_mask (`tensor`, optional): padding mask. size: [batch, N_seq, N_seq]. Defaults to None.
        """
        predict_R, predict_Trans = predict_T
        RotaionMatrix, translation = transformation
        delta_predict_Trans = rearrange(predict_Trans, 'b j t -> b j () t') - rearrange(predict_Trans, 'b i t -> b () '
                                                                                                       'i t')
        delta_Trans = rearrange(translation, 'b j t -> b j () t') - rearrange(translation, 'b i t -> b () i t')

        X_hat = torch.einsum('bikq, bjik->bijq', predict_R, delta_predict_Trans)
        X = torch.einsum('bikq, bjik->bijq', RotaionMatrix, delta_Trans)

        distance = torch.norm(X_hat-X, dim=-1)
        distance = torch.clamp(distance, max=self.clamp) * (1/self.z)

        if pdb_mask is not None:
            distance = distance * pdb_mask
        if padding_mask is not None:
            distance = distance * padding_mask

        FAPE_loss = torch.mean(distance)

        return FAPE_loss


def KL_loss(outputs, weight):
    kl_loss = 0.5 / outputs[0].size(1) * (1 + 2 * outputs[2] - outputs[1] ** 2 - torch.exp(outputs[2]) ** 2).sum(1).mean()
    return kl_loss * weight


def makeW(r1: float, r2: float, r3: float, r4: float = 0) -> torch.Tensor:
    """
    matrix involved in quaternion rotation
    """
    W = torch.Tensor(
        [
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    ).type(cfg.FloatTensor)
    return W


def makeQ(r1: float, r2: float, r3: float, r4: float = 0) -> torch.Tensor:
    """
    matrix involved in quaternion rotation
    """
    Q = torch.Tensor(
        [
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    ).type(cfg.FloatTensor)
    return Q


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def randomSeed(random_seed):
    """Given a random seed, this will help reproduce results across runs"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def get_activation(activation, slope=0.2):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "lrelu":
        return nn.LeakyReLU(slope)
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise KeyError(activation)


def clearCache():
    torch.cuda.empty_cache()
