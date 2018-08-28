import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def compute_mask(t, padding_idx=0):
    """
    compute mask on given tensor t
    :param t:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(t, padding_idx).float()
    return mask


# Trim a tensor by the length of the max thing
trim = lambda x : x[:, :(x.shape[1] - torch.min(torch.sum(x.eq(0).long(), dim=1))).item()]
