import torch
import torch.nn as nn
import numpy as np


class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self, reduction="batchmean"):
        super().__init__(reduction=reduction)

    def forward(self, y, t):
        # KLDivergenceは確率分布の対数を入力とするため、softmax後にlogを取る
        y = nn.functional.log_softmax(y, dim=1)
        loss = super().forward(y, t)

        return loss


class KLDivLossWithLogitsForVal(nn.KLDivLoss):
    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.numpy())
        self.label_list.append(t.numpy())

    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = (
            super().forward(torch.from_numpy(log_prob), torch.from_numpy(label)).item()
        )
        self.log_prob_list = []
        self.label_list = []

        return final_metric
