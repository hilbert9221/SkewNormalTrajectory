import torch
from torch.distributions import Categorical, Normal, Independent
import sys
sys.path.append('..')
from skew_normal_class import DiagSkewNormal
from my_mixture_same_family import MixtureSameFamily


def gaussian_mixture_loss(W_pred, S_trgt, n_stop, dist='normal'):
    r"""Batch gaussian mixture loss"""
    # NMV(C*K) -> NVM(C*K)
    W_pred = W_pred.transpose(1, 2).contiguous()

    temp = S_trgt.chunk(chunks=n_stop, dim=1)
    W_trgt_list = [i.mean(dim=1) for i in temp]
    W_pred_list = W_pred.chunk(chunks=n_stop, dim=-1)

    loss_list = []
    deltas = []
    for i in range(n_stop):
        # NVMC
        # [1, node, K, 5]: 1 for component, 4 for mean and variance
        W_pred_one = W_pred_list[i]
        W_trgt_one = W_trgt_list[i]
        # diag covariance matrix
        if dist == 'normal':
            mix = Categorical(torch.nn.functional.softmax(W_pred_one[:, :, :, 4], dim=-1))
            comp = Independent(Normal(W_pred_one[:, :, :, 0:2], W_pred_one[:, :, :, 2:4].exp()), 1)
        elif dist == 'skew':  
            mix = Categorical(torch.nn.functional.softmax(W_pred_one[:, :, :, -1], dim=-1))
            comp = Independent(DiagSkewNormal(logits=W_pred_one[..., :-1]), 0)
        else:
            raise ValueError('Invalid distribution type')
        gmm = MixtureSameFamily(mix, comp)
        loss_list.append(-gmm.log_prob(W_trgt_one))

    loss = torch.cat(loss_list, dim=0)
    return loss.mean()


def mse_loss(S_pred, S_trgt, loss_mask, training=True, add_fde=True):
    r"""Batch mean square error loss"""
    # NTVC
    if isinstance(loss_mask, list):
        loss = (S_pred - S_trgt).norm(p=2, dim=3) ** 2
        losses = loss.chunk(chunks=len(loss_mask), dim=1)
        losses = [loss.mean(dim=1) * mask for loss, mask in zip(losses, loss_mask)]
        if training:
            for i in range(len(losses)):
                losses[i][losses[i] > 1] = 0
        return sum([loss.mean() for loss in losses]) / len(losses)
    else:
        loss = (S_pred - S_trgt).norm(p=2, dim=3) ** 2
        loss = loss.mean(dim=1) * loss_mask
        if training:
            loss[loss > 1] = 0
        return loss.mean()
