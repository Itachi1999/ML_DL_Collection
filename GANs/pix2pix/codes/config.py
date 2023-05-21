import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

both_transform = A.Compose(
    [A.Resize(width=256, height=256),
     A.HorizontalFlip(p=0.5),
     A.Normalize(mean=[0.5, 0.5, 0.5], std=[
         0.5, 0.5, 0.5], max_pixel_value=255,),
     ToTensorV2(),],
    additional_targets={"image0": "image"},
)

ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
LR = 2e-4
DATA_DIR = 'D:/Datasets/pix2pix/maps'
BATCH_SIZE = 16
MDL_CKPT_PATH = 'ckpt/maps/'
MDL_FILENAME = 'pix2pix_{epoch}-{GLoss:.2f}-{DLoss:.2f}'
LOG_SAVE_DIR = 'log_dir/'
EXP_NAME = 'pix2pixMaps/'
VERSION = 0
