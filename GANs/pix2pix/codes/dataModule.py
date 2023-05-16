import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import glob
import torch.multiprocessing as mp
from pandas.core.common import flatten
from PIL import Image


class customDataset(Dataset):
    def __init__(self, img_pth, seg_pth, transform=None) -> None:
        super().__init__()
        self.img_pth = img_pth
        self.seg_path = seg_pth
        self.transform = transform

        self.train_imgs_paths = []
        self.train_seg_paths = []

        # Listing the train image and segmentaion path lists
        for data_pth in glob.glob(img_pth + '\\*'):
            self.train_imgs_paths.append(glob.glob(data_pth + '\\*'))
        self.train_imgs_paths = list(flatten(self.train_imgs_paths))
        self.train_imgs_paths = [a.replace('\\', '/')
                                 for a in self.train_imgs_paths]

        for data_pth in glob.glob(seg_pth + '\\*'):
            self.train_seg_paths.append(glob.glob(data_pth + '\\*'))
        self.train_seg_paths = list(flatten(self.train_seg_paths))
        self.train_seg_paths = [a.replace('\\', '/')
                                for a in self.train_seg_paths]

    def __len__(self):
        return len(self.train_imgs_paths)

    def __getitem__(self, index):
        img_path = self.train_imgs_paths[index]
        seg_path = self.train_seg_paths[index]
        # label = img_path.split('/')[-2]

        img = self.custom_pil_loader(img_path)
        seg = self.custom_pil_loader(seg_path)

        if self.transform is not None:
            img = self.transform(img)
            seg = self.transform(seg)
            # NOTE: MAKE THIS RIGHT

        return seg, img  # x, y

    def custom_pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()
            return img


class FNAC_seg_pair_dm(pl.LightningDataModule):
    def __init__(
            self, img_pth, seg_pth, num_workers=mp.cpu_count() // 2, seed=42) -> None:
        super().__init__()

        self.img_root_pth = img_pth
        self.seg_root_pth = seg_pth
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
