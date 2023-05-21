import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.multiprocessing as mp
from torch.utils.data import random_split
import torch


class FNAC_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size=16,
        transform=None,
        num_workers=mp.cpu_count(),
        seed=42,
        augmented=False,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.bs = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.seed = seed

        if augmented:
            self.split_ln = [216, 72, 72]
        else:
            self.split_ln = [112, 37, 37]

        self.generator = torch.Generator().manual_seed(seed=seed)

    def setup(self, stage: str) -> None:
        entire_ds = ImageFolder(
            root=self.root_dir,
            transform=self.transform
        )

        self.train_ds, self.val_ds, self.test_ds = random_split(
            entire_ds, lengths=self.split_ln, generator=self.generator)

    def train_dataloader(self):
        train_dl = DataLoader(self.train_ds, batch_size=self.bs,
                              shuffle=True, num_workers=self.num_workers)

        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_ds, batch_size=self.bs,
                            shuffle=False, num_workers=self.num_workers)

        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_ds, batch_size=self.bs,
                             shuffle=False, num_workers=self.num_workers)

        return test_dl
