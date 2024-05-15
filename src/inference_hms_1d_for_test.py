import os
import gc
import sys
import copy
import pywt
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass, field
from pathlib import Path
from tqdm.notebook import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


MODE = "test"  # "test" or "val"
# MODE = "val"  # "test" or "val"
OUTPUT_FOR_ENSEMBLE = True
DEBUG = False  # False #True
DEBUG_SAMPLE_NUM = 100
KERNEL = Path("/kaggle/input/hms-harmful-brain-activity-classification").exists()

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
DATA = (
    Path("/kaggle/input/hms-harmful-brain-activity-classification")
    if KERNEL
    else Path("../hms-harmful-brain-activity-classification")
)
OUTPUT = Path("./") if KERNEL else Path("../submissions")
if not KERNEL:
    OUTPUT.mkdir(parents=True, exist_ok=True)
TMP = Path("./.tmp") if KERNEL else None
CKPTS = Path("") if KERNEL else Path("../checkpoints/")


print("========================")
print("mode :", MODE)
print("env  :", "KERNEL" if KERNEL else "LOCAL")
print("debug:", DEBUG)
print("========================")


if KERNEL:
    sys.path.append(os.path.join(os.getcwd(), "/kaggle/input/hms-hbac-src"))
else:
    sys.path.append(os.path.join(os.getcwd(), "../src"))
from mydatasets import HMSHBACSpecDataset, HMSSEDDataset, HMS1DDataset
from mydatasets.make_eeg_spectrograms import spectrogram_from_eeg
from augmentations import hms_spec_augmentations, hms_1D_augmentations
from metrics.kaggle_kl_div import score as calc_kl_div
import models as MODELS

if MODE == "val":
    # for fold v1
    csv_path = DATA / "train.csv" if KERNEL else DATA / "train_fold_irr_mark_v2.csv"
    # for fold v2
    # csv_path = DATA / "train.csv" if KERNEL else DATA / "train_fold_irr_mark_v5.csv"
    spec_dir_path = DATA / "train_spectrograms"
    eeg_dir_path = DATA / "train_eegs"
else:
    csv_path = DATA / "test.csv"
    spec_dir_path = DATA / "test_spectrograms"
    eeg_dir_path = DATA / "test_eegs"


@dataclass
class CFG:
    gpu: int = 0
    batch_size: int = 16
    n_workers: int = 1
    k_fold: int = 4
    checkpoints: List[str] = field(
        default_factory=lambda: [
            # LB=0.26: base モデル
            [
                "run-20240312_102706-0an6kdp8-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_finetune1e-4-nv10_e30_fold_0/",
                "run-20240312_134457-fq6753ul-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_finetune1e-4-nv10_e30_fold_1/",
                "run-20240312_134503-uk71e5ex-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_finetune1e-4-nv10_e30_fold_2/",
                "run-20240312_155203-xcob14sf-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_finetune1e-4-nv10_e30_fold_3/",
            ],
            # LB=0.25: downsampleモデル
            [
                "run-20240315_153328-zrrn4qhx-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_finetune_fold_0/",
                "run-20240315_153438-rqwbpbfb-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_finetune_fold_1/",
                "run-20240315_153442-0m77mgei-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_finetune_fold_2/",
                "run-20240315_153445-ujddmobr-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_finetune_fold_3/",
            ],
            # downsampleモデルのSeed=0, CV=0.252
            [
                "run-20240401_111123-vuot3hc1-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_finetune_fold_0/",
                "run-20240401_124844-1abnrom1-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_finetune_fold_1/",
                "run-20240401_142450-1ouy8vnm-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_finetune_fold_2/",
                "run-20240401_160306-ierghoep-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed0_finetune_fold_3/",
            ],
            # downsampleモデルのSeed=123, CV=0.247
            [
                "run-20240401_182516-c3nkrdyq-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_finetune_fold_0/",
                "run-20240401_123308-ycz11bsh-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_finetune_fold_1/",
                "run-20240401_140851-akcr7s7k-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_finetune_fold_2/",
                "run-20240401_154720-i5m93pz4-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_seed123_finetune_fold_3/",
            ],
            # downsampleモデルのbackbone = maxxvit_rmlp_small, CV=0.245
            [
                "run-20240401_111126-xo5scl7d-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_finetune_fold_0/",
                "run-20240401_125917-uay4ug48-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_finetune_fold_1/",
                "run-20240401_144726-czyaecih-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_finetune_fold_2/",
                "run-20240401_163800-65gz7sk5-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_finetune_fold_3/",
            ],
            # downsampleモデルのbackbone = tf_efficientnet_b4_ns, CV=0.269
            [
                "run-20240401_111124-9y3ixw3i-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_finetune_fold_0/",
                "run-20240401_124721-6uxqz0en-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_finetune_fold_1/",
                "run-20240401_142250-lsohp5jh-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_finetune_fold_2/",
                "run-20240401_160029-fdugba01-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e50_warmup_downwample_finetune_fold_3/",
            ],
            # downsampleモデルのbackbone = maxxvitv2_nano_rw_256, fold変え CV=0.239
            [
                "run-20240405_152214-wm5w0ew3-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_0/",
                "run-20240405_152227-th1rruwk-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_1/",
                "run-20240405_152228-xhwtkxba-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_2/",
                "run-20240405_165938-x123rriq-1D_cls_RTpIcS_LS_Wavenet-maxxvitv2n_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_3/",
            ],
            # downsampleモデルのbackbone = maxxvit_rmlp_small_rw_256, fold変え CV=0.240
            [
                "run-20240405_170053-kadak6et-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_0/",
                "run-20240405_165949-lbjfygkz-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_1/",
                "run-20240405_183700-wgsxjdxt-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_2/",
                "run-20240405_184943-9u133n8x-1D_cls_RTpIcS_LS_Wavenet-maxxvits_1e-3_standard_e50_warmup_downwample_foldV2_finetune_fold_3/",
            ],
            # downsampleモデルのbackbone = tf_efficientnet_b4_ns, fold変え CV=0.265
            [
                "run-20240405_185147-8gjfntzq-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_finetune_fold_0/",
                "run-20240405_202552-4q5x7wi1-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_finetune_fold_1/",
                "run-20240405_203742-j6v7ia0d-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_finetune_fold_2/",
                "run-20240405_195044-wteo0zct-1D_cls_RTpIcS_LS_Wavenet-effnetb4_1e-3_standard_e15_warmup_downwample_foldV2_finetune_fold_3/",
            ],
        ]
    )
    # checkpointsのリストごとにデータセットをどれ使うか指定する。
    # test:
    datasets: List = field(default_factory=lambda: [HMS1DDataset] * 9)
    # validation: fold v1
    # datasets: List = field(default_factory=lambda: [HMS1DDataset] * 6)
    # validation: fold v2
    # datasets: List = field(default_factory=lambda: [HMS1DDataset] * 3)
    # checkpointsのリストごとのweightを決定
    # test
    model_weight: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1, 1, 1])
    # validatiaon: fold v1
    # model_weight: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    # validation: fold v2
    # model_weight: List[int] = field(default_factory=lambda: [1, 1, 1])

    # Ensembleようにlogit等を保存する際のモデル名
    # test:
    model_name: List[str] = field(
        default_factory=lambda: [
            "wavenet_maxxvitv2n",
            "wavenet_maxxvitv2n_downsample",
            "wavenet_maxxvitv2n_downsample_seed0",
            "wavenet_maxxvitv2n_downsample_seed123",
            "wavenet_maxxvits_downsample",
            "wavenet_effnetb4_downsample",
            "wavenet_maxxvitv2n_downsample_foldV2",
            "wavenet_maxxvits_downsample_foldV2",
            "wavenet_effnetb4_downsample_foldV2",
        ]
    )
    # validation: fold v1
    # model_name: List[str] = field(
    #     default_factory=lambda: [
    #         "wavenet_maxxvitv2n",
    #         "wavenet_maxxvitv2n_downsample",
    #         "wavenet_maxxvitv2n_downsample_seed0",
    #         "wavenet_maxxvitv2n_downsample_seed123",
    #         "wavenet_maxxvits_downsample",
    #         "wavenet_effnetb4_downsample",
    #     ]
    # )
    # validation: fold v2
    # model_name: List[str] = field(
    #     default_factory=lambda: [
    #         "wavenet_maxxvitv2n_downsample_foldV2",
    #         "wavenet_maxxvits_downsample_foldV2",
    #         "wavenet_effnetb4_downsample_foldV2",
    #     ]
    # )

    # tta
    tta_type: List[str] = field(default_factory=lambda: [])
    # tta_type: List[str] = field(default_factory=lambda: ["Inversion"])
    # tta_type: List[str] = field(default_factory=lambda: ["Reverse"])
    # tta_type: List[str] = field(default_factory=lambda: ["ChannelSwap"])
    # tta_type: List[str] = field(default_factory=lambda: ["Inversion", "Reverse", "ChannelSwap"])

    csv_path: Path = csv_path
    spec_dir_path: Path = spec_dir_path
    eeg_dir_path: Path = eeg_dir_path

    # postprocess
    scale_coeff: float = 0


def allkeys(x):
    for key, value in x.items():
        yield key
        if isinstance(value, dict):
            for child in allkeys(value):
                yield key + "." + child


def check_dotlist(cfg, dotlist):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_keys = list(allkeys(cfg_dict))
    dotlist_dict = OmegaConf.to_container(dotlist, resolve=True)
    dotlist_keys = list(allkeys(dotlist_dict))

    for d_key in dotlist_keys:
        assert d_key in cfg_keys, f"{d_key} dosen't exist in config file."


def load_configs(checkpoints):
    # latesub: checkpointsをディレクトリを指定するようにしたため、.ckptと.yamlを両方検索するように変更
    configs_list = []
    checkpoints_list = []
    for ckpts_dir in checkpoints:
        configs = []
        ckpt_fold = []
        for c in ckpts_dir:
            c = CKPTS / c
            conf_path = c / "train_config.yaml"
            conf = OmegaConf.load(conf_path)
            ckpt = list(c.glob("*.ckpt"))[0]
            ckpt_fold.append(c / ckpt.name)
            configs.append(conf)
        configs_list.append(configs)
        checkpoints_list.append(ckpt_fold)

    return checkpoints_list, configs_list


# def load_configs(checkpoints):
#     configs_list = []
#     checkpoints_list = []
#     for ckpts in checkpoints:
#         configs = []
#         ckpt_fold = []
#         for c in ckpts:
#             c = CKPTS / c
#             conf_path = c.parent / "train_config.yaml"
#             conf = OmegaConf.load(conf_path)
#             ckpt_fold.append(c)
#             configs.append(conf)
#         configs_list.append(configs)
#         checkpoints_list.append(ckpt_fold)

#     return checkpoints_list, configs_list


def get_transforms(config):
    # transform
    height = config.height
    width = config.width
    augment_args = config.transforms
    transforms = hms_1D_augmentations(augment_args=augment_args)
    return transforms


# spectrogram_from_eegの処理がdataloader内で行うと遅いので、別で行う。
def create_eeg_spec(denoise_wavelet=None):
    paths_eegs = list(GLOBAL_CFG.eeg_dir_path.iterdir())
    print(f"There are {len(paths_eegs)} EEG spectrograms")
    all_eegs = {}
    counter = 0
    save_dir = TMP / f"EEG_Spectrograms"
    save_dir.mkdir(parents=True, exist_ok=True)
    for file_path in tqdm(paths_eegs):
        file_path = str(file_path)
        eeg_id = file_path.split("/")[-1].split(".")[0]
        save_path = save_dir / f"{eeg_id}.npy"
        eeg_spectrogram = spectrogram_from_eeg(
            file_path, denoise_wavelet=denoise_wavelet, display=counter < 1
        )
        # all_eegs[int(eeg_id)] = eeg_spectrogram
        np.save(save_path, eeg_spectrogram)
        counter += 1
    return save_path.parent


def load_weight(checkpoint, net):
    ckpt = torch.load(checkpoint, map_location=f"cuda:{GLOBAL_CFG.gpu}")["state_dict"]
    ckpt = {k[k.find(".") + 1 :]: v for k, v in ckpt.items()}
    missing_keys, unexpected_keys = net.load_state_dict(ckpt, strict=False)
    print(f"\nload checkpoint: {checkpoint}\n")
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("====================================")
        print("missing_keys:", missing_keys)
        print("unexpecte_keys:", unexpected_keys)
        print("====================================")
    return net


def get_models_from_checkkpoints(ckpt_paths, configs):
    model_list = []
    for ckpt, config in zip(ckpt_paths, configs):
        model_args = copy.deepcopy(config["model"])
        model_args.load_checkpoint = None
        for key in model_args.args.keys():
            if "pretrained" in key:
                model_args.args[key] = False
        model = getattr(MODELS, model_args.name)(**model_args.args)
        model.to(GLOBAL_CFG.gpu)
        model = load_weight(ckpt, model)
        model.eval()
        model_list.append(model)
    return model_list


def get_dataloader(data_module, config, transforms, batch_size=1, num_workers=4):
    dataset_args = copy.deepcopy(config["dataset"])
    dataset_args.csv_path = str(GLOBAL_CFG.csv_path)
    dataset_args.spec_dir_path = str(GLOBAL_CFG.spec_dir_path)
    dataset_args.eeg_dir_path = str(GLOBAL_CFG.eeg_dir_path)
    eeg_spec_dir_path = dataset_args.get("eeg_spec_dir_path", None)
    # TODO: modelごとに異なるEEG_specを使う場合は修正の必要あり
    if eeg_spec_dir_path is not None and KERNEL:
        dataset_args.eeg_spec_dir_path = str(TMP / "EEG_Spectrograms")

    # 推論時は事前作成済みのデータは使わないのでオフにする(configから削除したのでコメントアウト)
    # dataset_args.spec_npy_path = None

    # pseudo_label用に学習した場合は、Falseにすることで、Testデータを推論するようにする。
    if hasattr(dataset_args, "for_pseudo_label"):
        dataset_args.for_pseudo_label = False
    if KERNEL or (MODE == "test"):
        dataset_args.fold = None

    # modeによらずtestモードのデータセットを作成。(csvファイルの全行を予測)
    dataset = data_module(mode="test", transforms=transforms, **dataset_args)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader


def tta_pred(net, input):
    preds = []
    if "Inversion" in GLOBAL_CFG.tta_type:
        y = net(-1 * input)
        preds.append(y["pred"])
    if "Reverse" in GLOBAL_CFG.tta_type:
        y = net(torch.flip(input, dims=[1]))
        preds.append(y["pred"])
    if "ChannelSwap" in GLOBAL_CFG.tta_type:
        b, t, c = input.shape
        mid = c // 2
        input = torch.cat((input[..., mid:], input[..., :mid]), dim=-1)
        y = net(input)
        preds.append(y["pred"])
    preds = torch.stack(preds, dim=0).mean(dim=0)
    return preds


def run_inference_loop(model_list, dataloader, transform=None):
    """test時のループ
    各foldのモデルで同じデータセットを予測し平均を取り、バッチ方向に連結。
    """
    pred_list = []
    logit_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            eeg = batch["eeg"].to(GLOBAL_CFG.gpu)
            kspec = batch.get("Kspec")
            if (transform is not None) and (kspec is not None):
                kspec = transform(kspec.to(GLOBAL_CFG.gpu))

            preds = []
            logits = []
            for net in model_list:
                if len(GLOBAL_CFG.tta_type) > 0:
                    pred = tta_pred(net, eeg)
                else:
                    y = net(eeg)
                    pred = y["pred"]
                logits.append(pred)  # n, b, c
                pred = F.softmax(pred, dim=1)
                preds.append(pred)  # n, b, c
            preds = torch.stack(preds, dim=0).mean(0)  # b, c
            logits = torch.stack(logits, dim=0).mean(0)  # b, c
            preds = preds.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            pred_list.append(preds)
            logit_list.append(logits)
            if DEBUG and i >= DEBUG_SAMPLE_NUM:
                break

    pred_arr = np.concatenate(pred_list)
    logit_arr = np.concatenate(logit_list)
    del pred_list, logit_list
    return {"pred_arr": pred_arr, "logit_arr": logit_arr}


def run_validatioan_loop(model_list, dataloader_list, transform=None):
    """validation時のループ
    各foldごとに別のデータセットを予測しすべてをバッチ方向に連結。
    """
    pred_list = []
    logit_list = []
    with torch.no_grad():
        for net, dataloader in zip(model_list, dataloader_list):
            for i, batch in enumerate(tqdm(dataloader)):
                eeg = batch["eeg"].to(GLOBAL_CFG.gpu)
                kspec = batch.get("Kspec")
                if (transform is not None) and (kspec is not None):
                    kspec = transform(kspec.to(GLOBAL_CFG.gpu))
                if len(GLOBAL_CFG.tta_type) > 0:
                    pred = tta_pred(net, eeg)
                else:
                    y = net(eeg)
                    pred = y["pred"]
                logit = pred
                pred = F.softmax(pred, dim=1)
                logit = logit.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                logit_list.append(logit)
                pred_list.append(pred)
                if DEBUG and i >= DEBUG_SAMPLE_NUM:
                    break

    pred_arr = np.concatenate(pred_list)
    logit_arr = np.concatenate(logit_list)
    del pred_list, logit_list
    return {"pred_arr": pred_arr, "logit_arr": logit_arr}


def calc_metric(pred_df, columns):
    solution = pred_df.loc[:, ["eeg_id"] + CLASSES]
    submission = pred_df.loc[:, ["eeg_id"] + columns].rename(
        columns={c: C for c, C in zip(columns, CLASSES)}
    )
    cv = calc_kl_div(
        solution=solution, submission=submission, row_id_column_name="eeg_id"
    )
    return cv


def save_sub(output_path, pred, df, sample_submission):
    pred_df = pd.DataFrame(pred, columns=CLASSES)
    pred_df = pd.concat([df[["eeg_id"]], pred_df], axis=1)
    if not DEBUG:  # sample_submissionの順番に合わせる処理
        pred_df = pd.merge(
            sample_submission[["eeg_id"]], pred_df, on="eeg_id", how="left"
        )
    pred_df.to_csv(output_path, index=False)
    # pred_df.head()


def postprocess(pred, scale_coef=0.05):
    """ソフトマックス関数の出力を0.5に近づける調整"""
    # 0.5からの距離に基づいて調整
    adjusted = pred + (0.5 - pred) * scale_coef  # 調整係数
    # 正規化して総和を1に保つ
    adjusted_normalized = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted_normalized


if __name__ == "__main__":

    GLOBAL_CFG = CFG()

    assert (
        len(GLOBAL_CFG.checkpoints)
        == len(GLOBAL_CFG.datasets)
        == len(GLOBAL_CFG.model_name)
        == len(GLOBAL_CFG.model_weight)
    )

    ckpts_list, configs_list = load_configs(GLOBAL_CFG.checkpoints)
    models_list = []
    for ckpts, configs in zip(ckpts_list, configs_list):
        models_list.append(get_models_from_checkkpoints(ckpts, configs))

    # TODO: モデルごとに必要なeeg_specを作れるようにする。(現状は一つのみ)
    # kernelの場合事前にeeg_specを作っておく。
    if KERNEL and hasattr(configs_list[0][0].dataset, "eeg_spec_dir_path"):
        suffix = configs_list[0][0].dataset.eeg_spec_dir_path.split("_")[-1]
        # パスの最後にデノイズ方法が書いてあればそれを使う。書いてない場合は使わない
        denoise_wavelet = suffix if suffix in pywt.wavelist() else None
        create_eeg_spec(denoise_wavelet=denoise_wavelet)

    pred = []
    logit = []
    for ckpts, configs, models, data_module in zip(
        ckpts_list, configs_list, models_list, GLOBAL_CFG.datasets
    ):
        # transformはmodel(5fold)ごとに一つ作成
        transforms = get_transforms(configs[0])
        if KERNEL or (MODE == "test"):  # kernel or テスト時は一つのdataloaderで良い。
            dataloader = get_dataloader(
                data_module,
                configs[0],
                transforms["audio_val"],
                batch_size=GLOBAL_CFG.batch_size,
                num_workers=GLOBAL_CFG.n_workers,
            )
            result = run_inference_loop(
                models, dataloader, transform=transforms["torch_val"]
            )
            pred_arr = result["pred_arr"]
            logit_arr = result["logit_arr"]
        else:  # validatiaon時はdataloaderをそれぞれ作る。
            dataloaders = []
            for config, model in zip(configs, models):
                dataloader = get_dataloader(
                    data_module,
                    config,
                    transforms["audio_val"],
                    batch_size=GLOBAL_CFG.batch_size,
                    num_workers=GLOBAL_CFG.n_workers,
                )
                dataloaders.append(dataloader)
            result = run_validatioan_loop(
                models, dataloaders, transform=transforms["torch_val"]
            )
            pred_arr = result["pred_arr"]
            logit_arr = result["logit_arr"]
            # validation時はdfの情報を持っておく
            dfs = []
            for dataloader in dataloaders:
                if DEBUG:
                    dfs.append(
                        dataloader.dataset.df.iloc[
                            : (DEBUG_SAMPLE_NUM + 1) * GLOBAL_CFG.batch_size
                        ]
                    )
                else:
                    dfs.append(dataloader.dataset.df)
            train_df = pd.concat(dfs).reset_index(drop=True)

        pred.append(pred_arr)
        logit.append(logit_arr)

    pred_average = np.average(pred, axis=0, weights=GLOBAL_CFG.model_weight)

    del models_list
    torch.cuda.empty_cache()
    gc.collect()

    if KERNEL or MODE == "test":
        df = pd.read_csv(GLOBAL_CFG.csv_path)
        smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

        # submissionの保存
        save_sub(OUTPUT / "submission.csv", pred_average, df, smpl_sub)

        # ensemble用に各モデルのprobとlogitを保存
        if OUTPUT_FOR_ENSEMBLE:
            for i, (p, l) in enumerate(zip(pred, logit)):
                save_sub(
                    OUTPUT / f"{GLOBAL_CFG.model_name[i]}_prob.csv", p, df, smpl_sub
                )
                save_sub(
                    OUTPUT / f"{GLOBAL_CFG.model_name[i]}_logit.csv", l, df, smpl_sub
                )

    else:
        columns = ["pred_" + c for c in CLASSES]
        pred_df = pd.DataFrame(pred_average, columns=columns)
        pred_df = pd.concat([train_df, pred_df], axis=1)
        cv = calc_metric(pred_df, columns)
        high_vote_cv = calc_metric(pred_df[pred_df["n_votes"] >= 10], columns)
        low_vote_cv = calc_metric(pred_df[pred_df["n_votes"] < 10], columns)
        print("CV_kldiv             : ", cv)
        print("CV_kldiv_high_votes  : ", high_vote_cv)
        print("CV_kldiv_low_votes   : ", low_vote_cv)

        if MODE == "val" and not DEBUG:
            pred_df.to_csv(OUTPUT / "prediction.csv", index=False)
            pred_df.head()

        # モデルごとの確率値とlogitを保存
        for i, (p, l) in enumerate(zip(pred, logit)):
            cols_p = ["prob_" + c for c in CLASSES]
            cols_l = ["logit_" + c for c in CLASSES]
            sub_prob_df = pd.DataFrame(p, columns=cols_p)
            sub_logit_df = pd.DataFrame(l, columns=cols_l)
            sub_prob_df = pd.concat([train_df, sub_prob_df], axis=1)
            sub_logit_df = pd.concat([train_df, sub_logit_df], axis=1)
            if MODE == "val" and not DEBUG:
                sub_prob_df.to_csv(
                    OUTPUT / f"prob_{GLOBAL_CFG.model_name[i]}_oof.csv", index=False
                )
                sub_logit_df.to_csv(
                    OUTPUT / f"logit_{GLOBAL_CFG.model_name[i]}_oof.csv", index=False
                )
                pred_df.head()
