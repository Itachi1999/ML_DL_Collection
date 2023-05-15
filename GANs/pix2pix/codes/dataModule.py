import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import os
import torch.multiprocessing as mp


class customDataset(Dataset):
    def __init__(self, img_pth, seg_pth, transform) -> None:
        super().__init__()
        self.img_pth = img_pth
        self.seg_path = seg_pth
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)


class FNAC_seg_pair_dm(pl.LightningDataModule):
    def __init__(
            self, img_pth, seg_pth, num_workers=mp.cpu_count() // 2, seed=42) -> None:
        super().__init__()

        self.img_pth = img_pth
        self.seg_pth = seg_pth
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
