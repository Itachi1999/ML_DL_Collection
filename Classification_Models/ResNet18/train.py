from codes.restNet18 import resNet18
from codes.dataLoader import make_everyThing
from codes.utils import lossPlot, folder_check
import codes.config as config
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Dataloader creation
train_dl, val_dl, test_dl = make_everyThing(
    root_dir=config.DATA_DIR, augmented=config.AUGMENTED,
    transform=config.TRANSFORM, batch_size=config.BATCH_SIZE)


# Model Definition
model = resNet18(num_classes=2, pretrained=config.PRETRAINED).to(
    config.ACCELERATOR)
optim = torch.optim.Adam(model.parameters(), lr=config.LR, betas=(0.5, 0.999))
criterion = torch.nn.CrossEntropyLoss()

# Summary Writer


def train(model, dataloader, device, optim, criterion):
    model.train()
    running_loss = 0
    counter = 0
    for idx, data in tqdm(enumerate(dataloader), desc="Training loop", total=np.ceil(112/dataloader.batch_size)):
        counter += 1
        imgs = data[0]
        label = data[1]
        imgs = imgs.to(device)
        label = label.to(device)
        optim.zero_grad()
        scores = model(imgs)
        loss = criterion(scores, label)
        loss.backward()  # Calculate the partial derivatives w.r.t weights and bias
        running_loss += loss.item()
        optim.step()  # Update the weights and biases w.r.t the optimizer

    training_loss = running_loss / counter
    return training_loss


def eval(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), desc="Validation Loop", total=np.ceil(37/dataloader.batch_size)):
            counter += 1
            imgs = data[0]
            label = data[1]
            imgs = imgs.to(device)
            label = label.to(device)
            scores = model(imgs)

            loss = criterion(scores, label)
            running_loss += loss.item()

    validation_loss = running_loss / counter
    return validation_loss


def training_loop(epochs, model, trainloader, testloader, device, optimizer, criterion):
    train_loss = []
    valid_loss = []
    the_epoch = -1
    min_val_loss = 9999.0000
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(
            model, trainloader, device, optimizer, criterion
        )
        valid_epoch_loss = eval(
            model, testloader, device, criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")

        folder_check(config.CKPT_DIR)

        if valid_epoch_loss < min_val_loss:
            min_val_loss = valid_epoch_loss
            torch.save(
                model, f'{config.CKPT_DIR}/epoch_{epoch}_valLoss_{valid_epoch_loss:.3f}.pt')
            the_epoch = epoch
            print(f"\nModel saved at epoch: {epoch + 1} \n")

        print(f"------ End of Epoch {epoch + 1} -------")

    return train_loss, valid_loss, min_val_loss, the_epoch


train_loss, valid_loss, min_val_loss, the_epoch = training_loop(epochs=config.MAX_EPOCHS, model=model, trainloader=train_dl,
                                                                testloader=val_dl, device=config.ACCELERATOR, optimizer=optim, criterion=criterion)

folder_check(f'results/{config.VERSION}')
folder_check(f'results/{config.PLOT_NAME}')

lossPlot(train_loss, valid_loss, filePath=f'results/{config.PLOT_NAME}',
         filename=config.EXP_NAME)


def test_model(model, dataloader, device):
    CM = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating on Testing Data"):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # file_name
            preds = torch.argmax(outputs.data, 1)
            CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])

        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        acc = np.sum(np.diag(CM)/np.sum(CM))
        sensitivity = tp/(tp+fn)
        precision = tp/(tp+fp)

        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM)
        print('- Sensitivity : ', (tp/(tp+fn))*100)
        print('- Specificity : ', (tn/(tn+fp))*100)
        print('- Precision: ', (tp/(tp+fp))*100)
        print('- NPV: ', (tn/(tn+fn))*100)
        print('- F1 : ', ((2*sensitivity*precision)/(sensitivity+precision))*100)
        print()

    return acc, CM


model = torch.load(
    f'{config.CKPT_DIR}/epoch_{the_epoch}_valLoss_{min_val_loss:.3f}.pt').to(config.ACCELERATOR)
acc, CM = test_model(model, test_dl, config.ACCELERATOR)

df_cm = pd.DataFrame(CM, index=range(2), columns=range(2))
plt.figure(figsize=(10, 7))
fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
plt.savefig(f'results/{config.PLOT_NAME}{config.EXP_NAME}_CM.png')
plt.close(fig_)


with open(f'results/{config.PLOT_NAME}result_{config.EXP_NAME}.txt', 'a') as f:
    f.write(f"Experiment Name: {config.EXP_NAME}")
    f.write(f"Version: {config.VERSION}")
    f.write(f"Validation_loss: {min_val_loss}")
