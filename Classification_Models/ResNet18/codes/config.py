import torch
import torchvision.transforms as transforms

PRETRAINED = True
LOG_DIR = 'log_dir/'
EXP_NAME = 'tra_augment'
VERSION = 'pretrained' if PRETRAINED else 'scratch'

'''
    [no_augment, tra_augment, gen_augment, tra_gen_augment]
'''
PLOT_NAME = f'{VERSION}/{EXP_NAME}/'
CKPT_DIR = f'ckpt/{VERSION}/{EXP_NAME}/'
CKPT_NAME = 'epoch={epoch}-val_loss={val_loss:.2f}-val_acc={val_acc:.2f}'

if EXP_NAME == 'no_augment':
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    DATA_DIR = 'D:/Datasets/FNAC_Total_Dataset_/images'
    AUGMENTED = False

elif EXP_NAME == 'tra_augment':
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    DATA_DIR = 'D:/Datasets/FNAC_Total_Dataset_/images'
    AUGMENTED = False

elif EXP_NAME == 'gen_augment':
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    DATA_DIR = 'D:/Datasets/FNAC_Total_Dataset_aug/'
    AUGMENTED = True

else:
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    DATA_DIR = 'D:/Datasets/FNAC_Total_Dataset_aug/'
    AUGMENTED = True


# Computation Details
ACCELERATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECISION = '16-mixed'
MIN_EPOCHS = 20
MAX_EPOCHS = 50
BATCH_SIZE = 16
LR = 3e-4
