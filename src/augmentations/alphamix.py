"""
hmsコンペ用alphaブレンドによるmixup
中央の10秒間のラベルを予測するコンペなので、中央から両端に行くに従ってalphaブレンドの割合を高くする。
中央のalphaブレンドの割合は低いのでラベルは変更しない。
"""

import numpy as np
import torch


def alphamix_data(x, alpha=0.4):
    """
    Applies alphamix to a sample
    Arguments:
        x {torch tensor} -- Input batch
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
    Returns:
        torch tensor  -- Mixed input
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).type_as(x).long()
    mid = x.size()[-1] // 2
    weight = torch.arange(mid, device=x.device, dtype=x.dtype) / mid * lam
    weight = torch.cat([torch.flip(weight, dims=[0]), weight], dim=0)

    mixed_x = x * (1 - weight) + x[index, :] * weight
    return mixed_x
