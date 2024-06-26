import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1) # n_samples, n_in, n_time
        cla = self.nonlinear_transform(self.cla(x)) # n_samples, n_in, n_time
        x = torch.sum(norm_att * cla, dim=2) # n_samples, n_classes
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class TimmSED(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        # config=None,
        n_mels=128,
        pretrained=False,
        num_classes=24,
        in_channels=1,
    ):
        super().__init__()

        # self.config = config

        self.bn0 = nn.BatchNorm2d(n_mels)

        base_model = timm.create_model(
            base_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=in_channels,
        )

        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        self.in_features = base_model.num_features

        self.fc1 = nn.Linear(self.in_features, self.in_features, bias=True)
        self.att_block = AttBlockV2(self.in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input_data):
        # if self.config.in_channels == 3:
        #     x = input_data
        # else:
        #     x = input_data[:, [0], :, :] # (batch_size, 1, time_steps, mel_bins)
        x = input_data  # (b, ch, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = x.transpose(2, 3)  # (b, ch, mel_bins, time_steps)

        x = self.encoder(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=2)  # (b, ch, time_steps)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = x.transpose(1, 2)# (b, time_steps, ch)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2) # (b, ch, time_steps)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)

        output_dict = {
            "x": x,
            "clipwise_output": clipwise_output,
            "norm_att": norm_att,
            "segmentwise_output": segmentwise_output,
        }

        return output_dict
