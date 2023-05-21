import pytorch_lightning as pl
# from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch
from codes.datasets import MapDataset, customFNACDataset


class FNAC_seg_pair_dm(pl.LightningDataModule):
    def __init__(
            self, img_pth, seg_pth, num_workers=mp.cpu_count() // 2, seed=42, transform=None, batch_size=8
    ) -> None:
        super().__init__()

        self.img_root_pth = img_pth
        self.seg_root_pth = seg_pth
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transform
        self.bs = batch_size

    def setup(self, stage: str):
        self.ds = customFNACDataset(
            self.img_root_pth, self.seg_root_pth, transform=self.transform)

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dl = DataLoader(
            self.ds, batch_size=self.bs, shuffle=True, num_workers=self.num_workers, generator=generator
        )

    def val_dataloader(self):
        pass


class MapDataModule(pl.LightningDataModule):
    def __init__(self, root_dir,  num_workers=mp.cpu_count() // 2, seed=42, batch_size=16) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.train_dir = f"{root_dir}/train/"
        self.val_dir = f"{root_dir}/val/"
        self.num_workers = num_workers
        self.seed = seed
        self.bs = batch_size

    def setup(self, stage: str):
        self.train_ds = MapDataset(root_dir=self.train_dir)
        self.val_ds = MapDataset(root_dir=self.val_dir)

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        train_dl = DataLoader(self.train_ds,
                              batch_size=self.bs,
                              shuffle=True,
                              num_workers=self.num_workers,
                              generator=generator)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_ds,
                            batch_size=self.bs,
                            shuffle=False,
                            num_workers=self.num_workers,
                            )
        return val_dl
