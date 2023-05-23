from torch.utils.data import DataLoader, random_split
import torch
from torchvision.datasets import ImageFolder


def make_everyThing(root_dir=None, seed=42, augmented=False, transform=None, batch_size=16):
    entire_ds = ImageFolder(root_dir, transform=transform)
    generator = torch.Generator().manual_seed(seed)

    if augmented:
        split_ln = [216, 72, 72]
    else:
        split_ln = [112, 37, 37]

    train_ds, val_ds, test_ds = random_split(
        entire_ds, lengths=split_ln, generator=generator)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl