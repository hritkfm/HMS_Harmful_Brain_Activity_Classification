import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
import copy

torchvision.disable_beta_transforms_warning()
import torchvision.transforms as T
import torchvision.transforms.v2 as Tv2
from albumentations.pytorch.transforms import ToTensorV2
import audiomentations as AUDIO
from augmentations import audio as myAudio


def get_albu_transforms(configs):
    transform_list = []
    if configs is not None:
        for item in configs:
            transform_type = item["type"]
            params = item.get("params", {})
            if hasattr(A, transform_type):
                transform_class = getattr(A, transform_type)
                transform_list.append(transform_class(**params))
            else:
                raise ValueError(f"Unsupported transform type: {transform_type}")

    transform_list.append(ToTensorV2(transpose_mask=True, p=1.0))  # HWC -> CHW
    return A.Compose(transform_list)


def hms_spec_augmentations(height, width, augment_args):
    albu_train = get_albu_transforms(augment_args["train"])
    albu_val = get_albu_transforms(augment_args["val"])

    transform = {
        "albu_train": albu_train,
        "torch_train": T.Compose(
            [
                T.Resize((height, width), antialias=None),
                T.ConvertImageDtype(torch.float),
            ]
        ),
        "albu_val": albu_val,
        "torch_val": T.Compose(
            [
                T.Resize((height, width), antialias=None),
                T.ConvertImageDtype(torch.float),
            ]
        ),
    }

    return transform


def _get_transforms(module, args):
    transforms = []
    if args is None:
        return []
    args = copy.deepcopy(args)
    for arg in args:
        transform_type = arg.pop(0)
        if hasattr(module, transform_type):
            params = {arg[i]: arg[i + 1] for i in range(0, len(arg), 2)}
            transform_class = getattr(module, transform_type)
            transforms.append(transform_class(**params))
        # elif transform_type == "PinkNoiseSNR":
        elif hasattr(myAudio, transform_type):
            params = {arg[i]: arg[i + 1] for i in range(0, len(arg), 2)}
            method = getattr(myAudio, transform_type)
            transform_class = AUDIO.Lambda(transform=method, **params)
            transforms.append(transform_class)

    return transforms




def hms_1D_augmentations(augment_args):
    audio_train = _get_transforms(AUDIO, augment_args.train)
    audio_val = _get_transforms(AUDIO, augment_args.val)
    transform = {
        "audio_train": AUDIO.Compose(audio_train),
        "torch_train": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
            ]
        ),
        "audio_val": AUDIO.Compose(audio_val),
        "torch_val": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
            ]
        ),
    }

    return transform
