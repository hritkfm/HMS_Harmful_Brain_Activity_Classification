from timm.optim import Lookahead, RAdam
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)
from transformers import AdamW as BERTAdam
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .adamp import AdamP
from .cosine_annealing_warmup_restarts import CosineAnnealingWarmupRestarts
from .demon import DEMONSGD, DEMONAdam
from .sam import SAM
