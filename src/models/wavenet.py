import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wave_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_rates: int,
        kernel_size: int = 3,
    ):
        """
        WaveNet building block.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param dilation_rates: how many levels of dilations are used.
        :param kernel_size: size of the convolving kernel.
        """
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        )

        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.gate_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)
            )

        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(
                self.convs[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(
                self.filter_convs[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(
                self.gate_convs[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.zeros_(self.gate_convs[i].bias)

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x)
            res = res + x
        return res

class SEModule1D(nn.Module):
    def __init__(self, ch: int, ratio: int = 16) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(ch, ch // ratio)
        self.fc2 = nn.Linear(ch//ratio, ch)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        h = self.pooling(x).squeeze(dim=-1)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h).sigmoid().unsqueeze(dim=-1)
        return h * x

class WaveNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        kernel_size: int = 3,
        hidden_channels: list[int] = [8, 16, 32, 64],
        dilation_rates: list[int] = [12, 8, 4, 1],
        downsample: bool = False, # trueの場合1/2**len(hidden_channels)に縮小される
        use_SE_module: bool = False, # 各層でチャンネルの方向にアテンションをかける
    ):
        super(WaveNet, self).__init__()
        hidden_channels = [input_channels] + hidden_channels
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.append(
            Wave_Block(
                hidden_channels[i],
                hidden_channels[i + 1],
                dilation_rates[i],
                kernel_size,
                )
            )
            if use_SE_module:
                layers.append(
                        SEModule1D(hidden_channels[i+1], ratio=1),
                )
            if downsample:
                layers.extend(
                    [
                        nn.Conv1d(hidden_channels[i + 1], hidden_channels[i + 1], kernel_size=3, stride=2, bias=True),
                        nn.BatchNorm1d(hidden_channels[i + 1]),
                        nn.SiLU(inplace=True),
                        ]
                )

        self.model = nn.Sequential(*layers)
        # layers = [
                # Wave_Block(
                # hidden_channels[i],
                # hidden_channels[i + 1],
                # dilation_rates[i],
                # kernel_size,
                # )
        #     for i in range(len(hidden_channels) - 1)
        # ]
        # self.model = nn.Sequential(
        #     Wave_Block(input_channels, 8, 12, kernel_size),
        #     Wave_Block(8, 16, 8, kernel_size),
        #     Wave_Block(16, 32, 4, kernel_size),
        #     Wave_Block(32, 64, 1, kernel_size),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # -> (b, c, time)
        output = self.model(x)
        return output  # -> (b, 64, time)
