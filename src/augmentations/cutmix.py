"""
https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
"""
import numpy as np
import torch


def cutmix_data(x, y, alpha=1.0):
    """
    Applies mixup to a sample
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
        hard_cutmix {bool} -- how to mix label (default: {False})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the shuffle batch
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).type_as(x).long()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2

def cutmix_data_v2(x, *y, alpha=1.0):
    """
    Applies mixup to a sample
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
        hard_cutmix {bool} -- how to mix label (default: {False})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the shuffle batch
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).type_as(x).long()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    label = []
    for l in y:
        label.append(lam * l + (1 - lam) * l[index])
    return x, *label