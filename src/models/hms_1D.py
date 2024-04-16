import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.SED import TimmSED, interpolate, pad_framewise_output
from models.pooling import GeM, GeM1d
from models.models1d.wavegram import WaveNetSpectrogram
from models.models1d.resnet1d import BasicBlock
from models.wavenet import WaveNet
from models.SED import AttBlockV2
from augmentations.mixup import mixup_data_v2
from nnAudio.features import MelSpectrogram, CQT1992v2


class WaveNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        kernel_size: int = 3,
        output_size: int = 320,
        pooling_type: str = "avg",
        encoder_channel_wise: bool = True,
        hidden_channels: list[int] = [8, 16, 32, 64],
        downsample: bool = False,
        use_SE_module: bool =False,
    ):

        super().__init__()

        self.encoder_channel_wise = encoder_channel_wise
        if encoder_channel_wise:
            in_channels = 1

        self.model = WaveNet(in_channels, kernel_size, hidden_channels=hidden_channels, downsample=downsample, use_SE_module=use_SE_module)
        self.pooling_type = pooling_type
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(output_size)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(output_size)

    def global_pooling(self, x):
        if self.pooling_type == "avg_max":
            x_ave = self.global_avg_pooling(x)  # b, ch, out_size
            x_max = self.global_max_pooling(x)  # b, ch, out_size
            x = x_ave + x_max
        elif self.pooling_type == "avg":
            x = self.global_avg_pooling(x)  # b, ch, out_size
        else:
            raise NotImplementedError
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwward pass.

        parametes:
            x (torch.tensor): [B, Time, ch]
        """
        num_channels = x.size(-1)
        if self.encoder_channel_wise:
            z = []
            for idx in range(num_channels):
                x1 = self.model(x[..., idx : idx + 1])  # -> (b, 64, time)
                x1 = self.global_pooling(x1)  # -> (b, 64, output_size)
                z.append(x1)

            y = torch.stack(z, dim=1)  # -> (b, ch, 64, output_size)

        else:
            y = self.model(x)
            y = self.global_pooling(y)

        return y  # (b, ch, 64, output_size)


class WavegramEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int | tuple = 64,
        wave_layers: tuple = (10, 10, 10, 10, 10, 10, 10, 10),
        wave_block: str = "waveblock",  # "simplified",
        output_size: int = 320,
    ):
        super().__init__()

        self.out_channels = len(wave_layers)

        # WaveNetSpectrogramの出力はb, wave_layersの数、base_filtersの最後の値, output_sizeになる
        # separate_channel=Trueの場合は、チャネルごとにwave_blockの計算が行われるため、
        # 入力のチャネル数とwave_layersの長さは対応している必要がある。
        self.model = WaveNetSpectrogram(
            in_channels=in_channels,
            base_filters=base_filters,
            wave_layers=wave_layers,
            wave_block=wave_block,
            output_size=output_size,
            separate_channel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)  # -> (b, num_wave_layers, base_filters, output_size))
        return y


class Parallel1DCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernels: list[int] = [128, 64, 32, 16, 11, 7, 5, 3],
        hidden_channels: list[int] = [4, 8],
        output_size: int = 320,
        encoder_channel_wise: bool = True,
    ):
        """異なるカーネルサイズの1DCNNを並列に畳み込む

        Args:
            in_channels (int): 入力チャネル数。encoder_channel_wise=Trueのときは、強制的に1
            kernels (list[int], optional): _description_. Defaults to [128, 64, 32, 16, 11, 7, 5, 3].
            hidden_channels (int): 各Blockの出力チャネル数。最終出力チャンネルはhidden_channels[-1] * n_kernels
        """

        super().__init__()
        self.kernels = kernels
        self.in_channels = in_channels
        self.parallel_conv = (
            nn.ModuleList()
        )  # modelにパラメータを登録するにはModuleListを使用する必要がある
        self.encoder_channel_wise = encoder_channel_wise
        if self.encoder_channel_wise:
            self.in_channels = 1

        self.hidden_channels = [in_channels] + hidden_channels
        for i, kernel_size in enumerate(list(self.kernels)):
            block = []
            for i_block, plane in enumerate(self.hidden_channels[:-1]):
                in_c = self.in_channels if i_block == 0 else plane
                out_c = self.hidden_channels[i_block + 1]

                sub_block = BasicBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    stride=1,
                    groups=1,
                    downsample=False,
                    use_bn=True,
                    is_first_block=(i_block == 0),
                )
                block.append(sub_block)

            block = nn.Sequential(*block)
            self.parallel_conv.append(block)
        self.global_pooling = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        """
        Forwward pass.

        parametes:
            x (torch.tensor): [B, Time, ch]
        """
        num_channels = x.size(-1)
        x = x.transpose(1, 2)
        if self.encoder_channel_wise:
            z = []
            for j in range(num_channels):
                para_result = []
                for i in range(len(self.kernels)):
                    x1 = self.parallel_conv[i](
                        x[:, j : j + 1]
                    )  # (b, 1, time) -> (b, planes, time)
                    x1 = self.global_pooling(x1)  # -> (b, planes, output_size)
                    para_result.append(x1)
                para_result = torch.cat(
                    para_result, dim=1
                )  # -> (b, planes * n_kernels, output_size)
                z.append(para_result)

            y = torch.stack(z, dim=1)  # -> (b, ch, planes * n_kernels, output_size)
        else:
            para_result = []
            for i in range(len(self.kernels)):
                x1 = self.parallel_conv[i](x)
                x1 = self.global_pooling(x1)
                para_result.append(x1)
            para_result = torch.cat(
                para_result, dim=1
            )  # -> (b, planes * n_kernels, output_size)
            y = para_result.unsqueeze(
                dim=1
            )  # -> (b, 1, planes * n_kernels, output_size)
        return y
    

class MelspecEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        sr:int = 200,
        n_fft: int = 1024,
        n_mels: int = 64,
        hop_length: int = int(10000 / 256), # audioの長さをhop_lengthで割った値が出力のtime_stepになる
        win_length: int = 128,
        fmin: int = 0,
        fmax: int = 100,
        trainable_mel: bool = False,
        trainable_STFT: bool = False,
        output_size: tuple = (64, 256), # 出力サイズ (h, w)
        downsample_method: str = "bilinear", # ["cnn", "bilinear"] ダウンサンプルする際の方法
    ):
        super().__init__()

        self.spec = MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length, 
            fmin=fmin,
            fmax=fmax,
            trainable_mel=trainable_mel,
            trainable_STFT=trainable_STFT,
        )
        self.instance_norm = nn.InstanceNorm1d(num_features=in_channels, affine=False)
        self.epsilon = 1e-10

        self.out_height = output_size[0]
        self.out_width = output_size[1]
        if downsample_method == "cnn" and (n_mels // self.out_height) > 1:
            num_cnn = (n_mels // self.out_height) // 2 # stride2でdownsampleする回数
            layers = []
            for i in range(num_cnn):
                layers.extend(
                    [
                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=(2, 1), padding=1),
                        nn.BatchNorm2d(in_channels),
                        nn.SiLU(inplace=True),
                    ]
                )
            self.downsampleH = nn.Sequential(*layers)
        else:
            self.downsampleH = nn.Identity()

        

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, ch, Time]
        # mean = x.mean(dim=2, keepdim=True)
        mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        # std = x.std(dim=2, keepdim=True)
        std = x.std(dim=1, keepdim=True).std(dim=2, keepdim=True)
        output = (x - mean) / (std + self.epsilon)
        return output
    
    def min_max_norm(self, x: torch.Tensor) -> torch.Tensor:
        xmax = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        xmin = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        # xmax = x.max(dim=2, keepdim=True)[0]
        # xmin = x.min(dim=2, keepdim=True)[0]
        output = (x - xmin) / (xmax - xmin + self.epsilon)
        return output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Time, ch]
        # MelSpectrogramへの入力は[num_audio, len_audio]なのでchをバッチ方向に連結
        b, t, ch = x.size()
        x = x.permute((0, 2, 1))
        x = self.instance_norm(x)
        # x = self.normalize(x)
        # x = self.min_max_norm(x)
        x = x.reshape((-1, t))
        y = self.spec(x) # [b*ch, t] -> [b*ch, freq_bins, time_steps]
        # y = torch.log(y+self.epsilon) # db化
        _, freq_bins, time_steps = y.size()
        y = y.reshape((b, ch, freq_bins, time_steps))

        # downsample or resize
        y = self.downsampleH(y)
        y = F.interpolate(y, size=(self.out_height, self.out_width))
        return y # [B, Ch, freq_bins, time_steps]

class sed_extracter(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.in_features, self.in_features, bias=True)
        self.att_block = AttBlockV2(self.in_features, num_classes, activation="linear")


    def forward(self, x):
        # x: (b, ch, time_steps)
        x = x.transpose(1, 2)# -> (b, time_steps, ch)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2) # (b, ch, time_steps)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        return {
            "clipwise_output": clipwise_output, # 全体の予測結果(b, num_classes)
            "norm_att": norm_att, # 時間方向の重み(b, num_classes, time_steps)
            "segmentwise_output": segmentwise_output, # 時間ごとの予測結果(b, num_classes, time_steps),
        }
    

class Squeeze(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

def split_width_and_stack_batch(data, n_split = 5):
    # data: (b, c, h, w)
    # dataをwidth方向にn_split個に分割してバッチ方向に積み上げる
    b, c, h, w = data.shape
    remainder = w % n_split
    data = data[..., :w-remainder]
        
    data = data.view((b, c, h, n_split, (w-remainder) // n_split))
    data = data.permute((0, 3, 1, 2, 4)).contiguous()
    data = data.reshape(b * n_split, c , h, -1)
    return data

def split_width_and_stack_batch_1d(data, n_split = 5):
    # data: (b, c, w)
    # dataをwidth方向にn_split個に分割してバッチ方向に積み上げる
    b, c, w = data.shape
    remainder = w % n_split
    data = data[..., :w-remainder]
    data = data.view((b, c, n_split, (w-remainder) // n_split))
    data = data.permute((0, 2, 1, 3)).contiguous()
    data = data.reshape(b * n_split, c, -1)
    return data

def merge_stacked_batch_to_original(data, n_split = 5):
    # data: (b, c, h, w)
    # split_width_and_stack_batchで分割したデータを元に戻す
    # (batch方向n_split個をwidth方向に結合)
    b, c, h, w = data.shape
    data = data.view(-1, n_split, c, h, w)
    data = data.permute((0, 2, 3, 1, 4)).contiguous()
    data = data.reshape(-1, c, h, w * n_split)
    return data

class WeightedAverage(nn.Module):
    def __init__(self, num_channels, trainable_weight: bool = True):
        super(WeightedAverage, self).__init__()
        # 重みパラメータを初期化（num_channelsに対応する形状）
        if trainable_weight:
            self.weights = nn.Parameter(torch.ones(num_channels, 1))
        else:
            self.weights = torch.tensor([[0.05], [0.2], [0.5], [0.2], [0.05]])
        
    def forward(self, x): 
        # channel方向に加重平均を計算
        # xの形状: (b, c, w)
        # 重みを正規化（オプション）
        weights_normalized = torch.softmax(self.weights, dim=0)  # 重みを正規化
        
        # 入力xを(b, w, c)に変更して、チャネル方向での加重平均を取りやすくする
        x_transposed = x.transpose(1, 2)  # (b, c, w) -> (b, w, c)
        
        # 加重平均を計算
        # 結果の形状: (b, w, 1) - weightsをブロードキャストして乗算
        result = torch.matmul(x_transposed, weights_normalized)
        
        # 不要な次元を削除して(b, w)の形状にする
        return result.squeeze(2)
class HMS1DModel(nn.Module):

    def __init__(
        self,
        data_split_num: int = 1, # データを何分割するか。 1(分割しない) or 5
        data_split_ver: str = "ver1", # ver1:wavenet後に分割、ver2: wavenet入力前に分割(学習時は10s,val時はバッチにスタック)
        encoder_type: str = "wavenet",
        encoder_pooling_type: str = "avg",  # "avg", "avg_max"
        encoder_channel_wise: bool = True,
        encoder_output_size: int = 320,
        feature_type: str = "half",  # "half", "standart", "double"
        extracter_type: str = "pooling",  # "pooling", "cnn2d"
        extracter_backbone: str = "resnet18d",
        extracter_pretrained: bool = False,
        extracter_dropout: float = 0.0,
        extracter_channel_wise: bool = False,  # チャンネルごとにモデルに入力する
        extracter_stack_diff_channel: bool = False, # LL-LPなどの差分情報をチャネルに追加する。
        head_type: str = "linear", # "linear"
        wavenet_params: dict | None = None,
        melspec_params: dict | None = None,
        use_sed_module: bool = False,
        p_manifold_mixup: int = 0,
        num_classes: int = 6,
    ):
        super().__init__()

        self.data_split_num = data_split_num
        self.data_split_ver = data_split_ver
        self.num_classes = num_classes
        self.p_manifold_mixup = p_manifold_mixup
        self.extracter_stack_diff_channel = extracter_stack_diff_channel
        self.feature_type = feature_type

        if feature_type == "half":
            self.input_channels = 8
            averaged_every_n_channels = 2
        elif feature_type == "standard":
            self.input_channels = 16
            averaged_every_n_channels = 4
        elif feature_type == "double":
            self.input_channels = 24
            averaged_every_n_channels = 4
        else:
            print(
                "feature_type must be 'half' or 'standard' or 'double'. Not",
                feature_type,
            )
            raise NotImplementedError

        self.averaged_every_n_channels = averaged_every_n_channels

        # encoderは出力が(B, input_channels, 64, encoder_output_size)になるようにする。
        if encoder_type == "wavenet":
            hidden_channels = [8, 16, 32, 64]
            downsample = False
            use_SE_module = False
            if wavenet_params is not None:
                hidden_channels = wavenet_params.get("hidden_channels", [8, 16, 32, 64])
                downsample = wavenet_params.get("downsample", False)
                use_SE_module = wavenet_params.get("use_SE_module", False)

            self.encoder = WaveNetEncoder(
                in_channels=self.input_channels,
                output_size=encoder_output_size,
                pooling_type=encoder_pooling_type,
                encoder_channel_wise=encoder_channel_wise,
                hidden_channels=hidden_channels,
                downsample=downsample,
                use_SE_module=use_SE_module,
            )
            encoder_output_height = hidden_channels[-1]
        elif encoder_type == "wavegram":
            self.encoder = WavegramEncoder(
                in_channels=1,
                base_filters=64,
                output_size=encoder_output_size,
                wave_layers=[10] * self.input_channels,
            )
            encoder_output_height = 64

        elif encoder_type == "parallelcnn":
            hidden_channels = [4, 8, 16, 16]
            kernels = [19, 13, 9, 5]
            self.encoder = Parallel1DCNNEncoder(
                in_channels=self.input_channels,
                hidden_channels=hidden_channels,
                kernels=kernels,
                output_size=encoder_output_size,
                encoder_channel_wise=True,
            )
            encoder_output_height = hidden_channels[-1] * len(kernels)
        elif encoder_type == "melspec":
            n_fft = 1024,
            n_mels = 128,
            hop_length = 10000 // encoder_output_size,
            win_length = 128,
            trainable_mel = False,
            trainable_STFT = False,
            output_height = n_mels
            if melspec_params is not None:
                n_fft = melspec_params.get("n_fft", 1024)
                n_mels = melspec_params.get("n_mels", 128)
                win_length = melspec_params.get("win_length", 128)
                trainable_mel = melspec_params.get("trainable_mel", False)
                trainable_STFT = melspec_params.get("trainable_STFT", False)
                output_height = melspec_params.get("output_height", n_mels)

            # encoder_output_height = 64
            encoder_output_height = output_height

            self.encoder = MelspecEncoder(
                in_channels=self.input_channels,
                sr=200,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length= hop_length,
                win_length= win_length,
                fmin=0,
                fmax=20,
                trainable_mel=trainable_mel , #False,
                trainable_STFT=trainable_STFT, #False,
                output_size=(encoder_output_height, encoder_output_size),
                downsample_method="bilinear"
            )
        else:
            print("encoder_type must be 'wavenet' or 'wavegram' or 'parallelcnn'. Not", encoder_type)
            raise NotImplementedError

        self.extracter_type = extracter_type
        self.extracter_channel_wise = extracter_channel_wise

        img_height = encoder_output_height * (
            self.input_channels // averaged_every_n_channels
        )
        img_width = encoder_output_size
        self.img_size = (img_height, img_width)

        if extracter_type == "pooling":
            self.extracter_channel_wise = False
            num_features = img_height # poolingの場合は画像の高さ方向がそのままheadの入力特徴となる
            self.extracter = nn.Sequential(nn.AdaptiveAvgPool1d(1), Squeeze(dim=-1))

        elif extracter_type == "cnn2d":
            net = timm.create_model(
                model_name=extracter_backbone,
                pretrained=extracter_pretrained,
                num_classes=0,
                # global_pool="",
                in_chans=1 + int(self.extracter_stack_diff_channel),
            )

            if self.extracter_channel_wise:
                ratio = self.input_channels // averaged_every_n_channels
                num_features = net.num_features * ratio
            else:
                num_features = net.num_features

            # modelごとに特徴抽出までをextracterに格納。
            if "swin" in extracter_backbone:
                net.head = nn.Sequential(*list(net.head.children())[:-2])
                layers = list(net.children())
                self.img_size = (256, 256)
            elif "convnext" in extracter_backbone:
                net.head = nn.Sequential(*list(net.head.children())[:-1])
                layers = list(net.children())
            elif "maxvit" in extracter_backbone:
                net.head = nn.Sequential(*list(net.head.children())[:-2])
                layers = list(net.children())
                # self.img_size = (256, 256)
                self.img_size = (384, 384)
            elif "maxxvit" in extracter_backbone:
                net.head = nn.Sequential(*list(net.head.children())[:-2])
                layers = list(net.children())
                self.img_size = (256, 256)
            elif "coatn" in extracter_backbone:
                net.head = nn.Sequential(*list(net.head.children())[:-2])
                layers = list(net.children())
                self.img_size = (224, 224)
            else:
                layers = list(net.children())[:-2]
                layers.extend([GeM(), Squeeze(dim=(2, 3))])
            self.extracter = nn.Sequential(
                *layers,
                # GeM(),
                # Squeeze(dim=(2, 3)),
            )
        else:
            raise NotImplementedError
        self.encoder_output_height = encoder_output_height
        self.resize = T.Resize(self.img_size, antialias=True)
        
        self.use_sed_module = use_sed_module
        if self.use_sed_module:
            sed_in_features = self.img_size[0]
            self.sed = sed_extracter(in_features=sed_in_features, num_classes=num_classes)

        if self.data_split_num > 1:
            self.weighted_average = WeightedAverage(data_split_num) # 予測クラスを加重平均：精度悪化
            # num_features *= self.data_split_num # mlpで混ぜる。

        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(extracter_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, target=None, target_sed=None):
        """
        Forwward pass.

        parametes:
            x (torch.tensor): [B, Time, ch]
            target (torch.tensor): [B, num_classes]
            target_sed (torch.tensor): [B, Time, num_classes]
        """
        orig_bs = x.size(0)
        if self.data_split_num == 5 and self.data_split_ver == "ver2":
            if self.training: # 学習時はランダムに10秒分を取得
                start = torch.randint(8_000, (1,))
                x = x[:, start:start + 2_000]
            else: # val時は10秒ごとにバッチ方向にスタックする。
                x = x.transpose(1,2) # b, ch , time
                x = split_width_and_stack_batch_1d(x, n_split=5) # b*5, ch, time/5
                x = x.transpose(1,2) # b*5, time/5, ch


        x = self.encoder(x)  # -> b, ch, height, width(Time)

        # manifold_mixupの処理
        p = torch.rand(1)
        if self.training and p < self.p_manifold_mixup:
            x, target, target_sed = mixup_data_v2(x, target, target_sed, alpha=2.0)

        result_sed = None 
        # データをWidth方向に分割しバッチ方向に積み上げる処理
        if self.data_split_num > 1 and self.data_split_ver == "ver1":
            x = split_width_and_stack_batch(x, n_split=self.data_split_num)

        # nチャンネルごとに平均を計算する処理
        b, c, h, w = x.size()
        if self.averaged_every_n_channels > 1:
            x = x.view(b, -1, self.averaged_every_n_channels, h, w)
            x = torch.mean(x, dim=2)  # -> (b, ch // n, 64, output_size )

        # channel次元のデータをheight方向に積み上げ、extracterで特徴抽出
        new_c = x.size(1)
        if self.extracter_type == "pooling":
            x = x.view(b, 1, new_c * h, w)
            x = x.squeeze(1)  # -> b, height, width(Time)
            if self.use_sed_module:
                result_sed = self.sed(x)["segmentwise_output"]
            x = self.extracter(x)
        else:
            if self.extracter_channel_wise:
                x = [self.extracter(self.resize(x[:, i : i + 1])) for i in range(new_c)]
                x = torch.cat(x, dim=1)
            else:
                x = x.view(b, 1, new_c * h, w)
                x = self.resize(x)
                if self.use_sed_module:
                    result_sed = self.sed(x.squeeze(1))["segmentwise_output"]
                if self.extracter_stack_diff_channel and self.feature_type != "double":
                    # abs(LL-LP), abs(LL-RL), abs(RP-LP), abs(RP-RL)のチャネルを追加("half" or "standard"のみ)
                    LL = x[:,:,:self.encoder_output_height]
                    LP = x[:,:,self.encoder_output_height:self.encoder_output_height*2]
                    RL = x[:,:,self.encoder_output_height*2:self.encoder_output_height*3]
                    RP = x[:,:,self.encoder_output_height*3:self.encoder_output_height*4]
                    LL_LP = torch.abs(LL-LP)
                    LL_RL = torch.abs(LL-RL)
                    RP_LP = torch.abs(RP-LP)
                    RP_RL = torch.abs(RP-RL)
                    diff = torch.cat([LL_LP, LL_RL, RP_LP, RP_RL], dim=2)
                    x = torch.cat([x, diff], dim=1)
                x = self.extracter(x)


        # バッチ方向に積み上げたデータをもとに戻す
        if self.data_split_num > 1:
            if self.data_split_ver == "ver2":
                y = self.head(x) # -> b*data_split_num, classes
                if not self.training:
                    result_sed = y.view(orig_bs, self.data_split_num, self.num_classes)
                    y = self.weighted_average(result_sed)
                    result_sed = None
            # result_sed = result_sed.permute((0, 2, 1))
            elif self.data_split_ver == "ver1":
                x = x.view(orig_bs, self.data_split_num, -1) # b*data_split_num, numfeatures
                x = x.reshape(orig_bs, -1)
                y = self.head(x)

        else:
            y = self.head(x)

        return {
            "pred": y, 
            "pred_sed": result_sed, # default: None
            "target": target,
            "target_sed": target_sed,
            }


class HMS1DWaveModel(nn.Module):

    def __init__(
        self,
        feature_type="standard",  # "half", "standart", "double"
        pooling="avg",  # "avg", "avg_max"
    ):
        super().__init__()

        self.model = WaveNet()
        self.pooling = pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.dropout = 0.0

        if feature_type == "half":
            head_feature_num = 64 * 4
        elif feature_type == "standard":
            head_feature_num = 64 * 8
        elif feature_type == "double":
            head_feature_num = 64 * 12
        else:
            raise NotImplementedError

        self.head = nn.Sequential(
            nn.Linear(head_feature_num, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 6),
        )

    def flat_channel(self, x, step=4):
        # チャネル次元を行方向に積み上げる。stepごとに列方向に結合する。
        b, c, f, time = x.shape
        reshaped_blocks = [
            x[:, i : i + step].reshape(b, 1, step * f, time) for i in range(0, c, step)
        ]
        x = torch.cat(reshaped_blocks, dim=-1)  # -> (b, 1, 4*freq, Time*ch/4)
        return x

    def global_pooling(self, x):
        if self.pooling == "avg_max":
            x_ave = self.global_avg_pooling(x)  # b, ch, 1
            x_max = self.global_max_pooling(x)  # b, ch, 1
            x = x_ave + x_max
        elif self.pooling == "avg":
            x = self.global_avg_pooling(x)  # b, ch, 1
        else:
            raise NotImplementedError
        return x

    def forward(self, x):
        """
        Forwward pass.
        """
        num_channels = x.size(-1)
        z = []
        for idx in range(0, num_channels, 2):
            x1 = self.model(x[..., idx : idx + 1])  # b, ch, time
            x1 = self.global_pooling(x1)
            x1 = x1.squeeze(-1)  # b, ch
            x2 = self.model(x[..., idx + 1 : idx + 2])
            x2 = self.global_pooling(x2)
            x2 = x2.squeeze(-1)
            z.append(torch.mean(torch.stack([x1, x2]), dim=0))  # b, ch

        y = torch.cat(z, dim=1)  # b, n*ch
        y = self.head(y)  # b, 6

        return {"pred": y}
