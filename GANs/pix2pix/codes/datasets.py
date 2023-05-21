import numpy as np
import codes.config as config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from pandas.core.common import flatten
import glob


class customFNACDataset(Dataset):
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
        assert img_path.split('/')[-1] == seg_path.split('/')[-1]

        img = self.custom_pil_loader(img_path)
        seg = self.custom_pil_loader(seg_path)

        if self.transform is not None:
            transformed = self.transform(image=img, seg=seg)
            img = transformed['image']
            seg = transformed['seg']

        return seg, img  # x, y

    def custom_pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()
            return img


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(
            image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # input_image = config.transform_only_input(image=input_image)["image"]
        # target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("D:/Datasets/pix2pix/maps/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
