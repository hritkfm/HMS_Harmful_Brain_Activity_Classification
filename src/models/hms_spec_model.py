import torch
import torch.nn as nn
import timm


class HMSHBACSpecModel(nn.Module):

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        in_channels: int,
        num_classes: int,
        channel_flatting: bool = True,
        half_as_mask: bool = False,
    ):
        super().__init__()
        self.channel_flatting = channel_flatting
        self.half_as_mask = half_as_mask

        self.model = timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=int(in_channels * (1 + int(half_as_mask))),
        )

    def forward(self, x):
        # x: (b, ch, Hz, Time)
        if self.channel_flatting:
            b, c, hz, time = x.shape
            reshaped_blocks = [
                x[:, i : i + 4].reshape(b, 1, 4 * hz, time) for i in range(0, c, 4)
            ]
            x = torch.cat(reshaped_blocks, dim=-1)  # -> (b, 1, 4*Hz, Time*ch/4)
            if self.half_as_mask:
                new_time = x.shape[-1]
                x = torch.cat(
                    [x[..., : int(new_time // 2)], x[..., int(new_time // 2) :]], dim=1
                )
        else:
            if self.half_as_mask:
                # maskをデータのチャンネルの間に入れる。(データとmaskが互い違いになるようにする)
                c = x.shape[1]
                new_x = torch.zeros_like(x)
                new_x[:, ::2, :, :] = x[:, : c // 2, :, :]
                new_x[:, 1::2, :, :] = x[:, c // 2 :, :, :]
                x = new_x

        h = self.model(x)

        return {"pred":h}
