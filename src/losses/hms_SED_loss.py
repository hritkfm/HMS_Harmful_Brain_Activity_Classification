import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from losses.KLDivLoss import KLDivLossWithLogits

class HMSSEDLoss(nn.Module):
    def __init__(self, loss_weight = [1, 0.5, 0.05]):
        super().__init__()

        self.KLDivLoss = KLDivLossWithLogits()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_weight = loss_weight

    def forward(self, y, y_framewise, y_specwise, target, target_framewise):
        loss = self.KLDivLoss(y, target)
        loss_kspec = self.KLDivLoss(y_specwise, target)
        b, frame, c = y_framewise.shape
        y_framewise = y_framewise.reshape(-1, c)
        target_framewise = target_framewise.reshape(-1, c)
        mask = target_framewise.sum(dim=1)
        mask = mask.unsqueeze(1)
        frame_loss = self.bce(y_framewise, target_framewise)
        frame_loss = (frame_loss * mask).sum() / len(mask.nonzero())

        return loss * self.loss_weight[0] + loss_kspec * self.loss_weight[1] + frame_loss * self.loss_weight[2]
    
class HMSSED1DLoss(nn.Module):
    def __init__(self, auxiliary_bce_loss=False, sed_loss_type=None, loss_weight = [1, 0.5, 0.5]):
        super().__init__()

        self.KLDivLoss = KLDivLossWithLogits()
        if auxiliary_bce_loss:
            self.auxiliary_loss = nn.BCEWithLogitsLoss(reduction="mean")
        if sed_loss_type == "bce":
            self.sed_loss = nn.BCEWithLogitsLoss(reduction="none")
        elif sed_loss_type == "KLDiv":
            self.sed_loss = KLDivLossWithLogits(reduction="none")
            
        self.loss_weight = loss_weight

    def forward(self, y, y_sed, target, target_sed):
        # y_sed: "segmentwise_output": 時間ごとの予測結果(b, num_classes, time_steps)
        # target_sed: (b, 50, n_classes)
        loss = self.KLDivLoss(y, target)
        loss_auxiliary = 0
        loss_sed = 0
        
        if hasattr(self, "auxiliary_loss"):
            loss_auxiliary = self.auxiliary_loss(y, target)

        if hasattr(self, "sed_loss"):
            # sed_loss, y_sedとtarget_sedの大きさをあわせてからロスを計算。(target_sedが全て0の箇所はマスクする)
            b, time_steps, num_classes = target_sed.size()
            y_sed = F.interpolate(y_sed, size=time_steps, mode="nearest")
            y_sed = y_sed.transpose(1,2) # -> (b, time_steps, num_classes)
            y_sed = y_sed.reshape(-1, num_classes)
            target_sed = target_sed.view(-1, num_classes)
            mask = target_sed.sum(dim=1)
            mask = mask.unsqueeze(1)
            loss_sed = self.sed_loss(y_sed, target_sed)
            loss_sed = (loss_sed * mask).sum() / len(mask.nonzero()) # mask!=0のところのロスの平均を計算

        return loss * self.loss_weight[0] + loss_auxiliary * self.loss_weight[1] + loss_sed * self.loss_weight[2]