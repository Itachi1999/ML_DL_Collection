# import numpy as np
import matplotlib.pyplot as plt
import os
# from torchvision.utils import make_grid
# import torchvision


def lossPlot(train_loss, val_loss, filePath, filename):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label="Train Loss", color="orange")
    plt.plot(val_loss, label="Validation Loss", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(filePath + filename + '.png')


def folder_check(path: str):
    if not os.path.exists(path):
        os.mkdir(path=path)

    return
