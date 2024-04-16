import random
from typing import List

import albumentations as A
import numpy as np


class AugMix(A.ImageOnlyTransform):
    def __init__(
        self,
        transforms: List,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.transforms = transforms
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, img, **params):
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        img_mix = np.zeros_like(img, dtype="float32")
        for i in range(self.width):
            img_aug = img.copy()
            depth = (
                self.depth
                if self.depth > 0
                else np.random.randint(1, min(len(self.transforms), 4))
            )
            transform = A.Compose(random.choices(self.transforms, k=depth))
            img_aug = transform(image=img_aug)["image"]
            img_mix += ws[i] * img_aug

        img = ((1 - m) * img + m * img_mix).astype(img.dtype)

        return img

    def get_transform_init_args_names(self):
        return ("transform", "width", "depth", "alpha")
