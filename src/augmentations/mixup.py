"""
https://github.com/hongyi-zhang/mixup/blob/master/cifar/easy_mixup.py
"""

import numpy as np
import torch


def mixup_data(x, y, alpha=0.4, hard_mixup=False):
    """
    Applies mixup to a sample
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
        hard_mixup {bool} -- how to mix label (default: {False})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the shuffle batch
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).type_as(x).long()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_data_v2(x, *y, alpha=0.4, hard_mixup=False):
    """
    Applies mixup to a sample
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
        hard_mixup {bool} -- how to mix label (default: {False})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the shuffle batch
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).type_as(x).long()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    label = []
    for l in y:
        label.append(lam * l + (1 - lam) * l[index])

    return mixed_x, *label



def zebramix_data(x, y):
    """波形を交互に混ぜる処理
    https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/discussion/275335
    Args:
        x (torch.tensor): Input batch (b, time, ch)
        y (torch.tensor): Labels
    """
    index = torch.randperm(x.size()[0]).type_as(x).long()

    wave_new = torch.zeros_like(x)
    wave_new[:, 0::2] = x[:, 0::2]
    wave_new[:, 1::2] = x[index, 1::2]
    y_a, y_b = y, y[index]
    lam = 0.5
    return wave_new, y_a, y_b, lam