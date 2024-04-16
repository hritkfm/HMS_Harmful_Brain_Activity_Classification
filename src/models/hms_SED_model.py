import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SED import TimmSED, interpolate, pad_framewise_output
from models.pooling import GeM, GeM1d


class HMSSEDModel(nn.Module):

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        in_channels: int,
        num_classes: int,
        channel_flatting: bool = True,
        sed_model: str = "tf_efficientnet_b0",
    ):
        super().__init__()

        # kspec(SED)
        self.sed_model = TimmSED(
            base_model_name=sed_model,
            n_mels=512 if channel_flatting else 128,
            pretrained=pretrained,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        sed_features = self.sed_model.in_features
        self.time_span = int(50 * 0.5 / 2)
        self.pooling1d = GeM1d()

        # espec
        espec_model = timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=in_channels,
        )
        layers = list(espec_model.children())[:-2]
        self.espec_encoder = nn.Sequential(*layers)
        espec_features = espec_model.num_features
        self.pooling2d = GeM()

        self.head = nn.Linear(sed_features + espec_features, num_classes)
        self.channel_flatting = channel_flatting

    def flat_channel(self, x, step=4):
        # チャネル次元を行方向に積み上げる。stepごとに列方向に結合する。
        b, c, f, time = x.shape
        reshaped_blocks = [
            x[:, i : i + step].reshape(b, 1, step * f, time) for i in range(0, c, step)
        ]
        x = torch.cat(reshaped_blocks, dim=-1)  # -> (b, 1, 4*freq, Time*ch/4)
        return x

    def forward(self, x):
        # x: (b, ch, freq, Time)
        kspec, espec = x[:, :4], x[:, 4:]
        # kspec: (b, ch, freq, Time)
        # espec: (b, ch, freq, Time)

        if self.channel_flatting:
            kspec = self.flat_channel(kspec, step=4)
            espec = self.flat_channel(espec, step=4)

        # kspec(SED)
        kspec = kspec.transpose(2, 3)  # b, ch, time, freq
        b, ch, frames_num, freq = kspec.shape
        # kspec = F.interpolate(
        #     kspec,
        #     size=(frames_num*2, freq),
        #     align_corners=True,
        #     mode="bilinear",
        # )
        sed_output = self.sed_model(kspec)
        specwise_output = sed_output["clipwise_output"]  # b, cla
        framewise_output = sed_output["segmentwise_output"]  # b, cla, time
        framewise_feature = sed_output["x"]  # b, n_features, time

        # framewise_outputとframewise_featureはtime方向に1/32されているので元のサイズに戻す。
        framewise_output = framewise_output.transpose(1, 2)  # b, time, cla
        interpolate_ratio = frames_num // framewise_output.size(1)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_feature = framewise_feature.transpose(1, 2)  # b, time, n_features
        interpolate_ratio = frames_num // framewise_feature.size(1)
        framewise_feature = interpolate(framewise_feature, interpolate_ratio)
        framewise_feature = pad_framewise_output(framewise_feature, frames_num)

        # framewise_featureの中央50sを取得する。
        framewise_feature = framewise_feature[
            :, frames_num // 2 - self.time_span : frames_num // 2 + self.time_span
        ]  # b, 24, n_features
        # ff1 = torch.mean(framewise_feature, dim=1)
        framewise_feature = framewise_feature.transpose(1, 2)
        ff1 = self.pooling1d(framewise_feature).squeeze(-1)
        # ff2 = torch.max(framewise_feature, dim=1)[0]
        # framewise_feature = ff1 + ff2  # b, n_features
        framewise_feature = ff1

        # espec
        espec = self.espec_encoder(
            espec
        )  # b, n_features, freq, time: b, n_features, 4, 8
        # espec = torch.mean(espec, dim=2)  # (b, n_features, time)
        # espec = espec.transpose(1, 2)
        # e1 = torch.mean(espec, dim=1)
        e1 = self.pooling2d(espec).squeeze(-1).squeeze(-1)  # b, n_features
        # e2 = torch.max(espec, dim=1)[0]
        # espec = e1 + e2  # b, n_features
        espec = e1

        x = torch.cat([framewise_feature, espec], dim=1)  # b, n_features

        # x = self.head(self.bn(x))
        x = self.head(x)

        return {
            "pred": x,
            "kspec_framewise_output": framewise_output,
            "kspec_specwise_output": specwise_output,
        }
