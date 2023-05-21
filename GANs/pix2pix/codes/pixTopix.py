from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
# from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from codes.gen_model import Generator
from codes.disc_model import Discriminator
from torchvision.utils import save_image


class pixToPix(pl.LightningModule):
    def __init__(
        self, test_dl, bs=8, lr=2e-4, in_channels=3, l1_lambda=100, lambda_gp=10, betas=(0.5, 0.999),
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.test_dl = test_dl
        self.bs = bs
        self.lr = lr
        self.in_channels = in_channels
        self.l1_lambda = l1_lambda
        self.lambda_gp = lambda_gp
        self.betas = betas

        self.disc = Discriminator(in_channels=in_channels)
        self.gen = Generator(in_channels=in_channels)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def discLoss(self, x, y, y_fake):
        D_real = self.disc(x, y)
        D_real_loss = self.bce(
            D_real, torch.ones_like(D_real, device=self.device))
        D_fake = self.disc(x, y_fake.detach())
        D_fake_loss = self.bce(
            D_fake, torch.zeros_like(D_fake, device=self.device))
        D_loss = (D_real_loss + D_fake_loss) / 2

        return D_loss

    def genLoss(self, x, y, y_fake):
        D_fake = self.disc(x, y_fake.detach())
        G_fake_loss = self.bce(
            D_fake, torch.ones_like(D_fake, device=self.device))
        L1 = self.l1_loss(y_fake, y) * self.l1_lambda
        G_loss = G_fake_loss + L1

        return G_loss

    def training_step(self, batch, batch_idx):
        opt_disc, opt_gen = self.optimizers()
        x, y = batch
        y_fake = self.gen(x)

        # Training Discriminator
        D_loss = self.discLoss(x, y, y_fake)
        opt_disc.zero_grad()
        self.manual_backward(D_loss)
        opt_disc.step()

        # Training Generator
        G_loss = self.genLoss(x, y, y_fake)
        opt_gen.zero_grad()
        self.manual_backward(G_loss)
        opt_gen.step()

        # Logging Things
        if batch_idx % 8 == 0:
            # x_grid = torchvision.utils.make_grid(x, normalize=True)
            # y_grid = torchvision.utils.make_grid(y, normalize=True)
            y_fake_grid = torchvision.utils.make_grid(
                y_fake, normalize=True)

            # self.logger.experiment.add_image(
            #     "real_images", x_grid, global_step=self.global_step)
            # self.logger.experiment.add_image(
            #     "segmentation_masks", y_grid, global_step=self.global_step)
            self.logger.experiment.add_image(
                "fake_training_images", y_fake_grid, global_step=self.current_epoch)

        log_dict = {"D_Loss": D_loss, "G_Loss": G_loss}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            y_fake = self.gen(x)

            D_loss = self.discLoss(x, y, y_fake)
            G_loss = self.genLoss(x, y, y_fake)

        if batch_idx % 2 == 0:
            # x_grid = torchvision.utils.make_grid(x, normalize=True)
            # y_grid = torchvision.utils.make_grid(y, normalize=True)
            y_fake_grid = torchvision.utils.make_grid(
                y_fake, normalize=True)

            # self.logger.experiment.add_image(
            #     "real_images", x_grid, global_step=self.global_step)
            # self.logger.experiment.add_image(
            #     "segmentation_masks", y_grid, global_step=self.global_step)
            self.logger.experiment.add_image(
                "fake_val_images", y_fake_grid, global_step=self.current_epoch)

        log_dict = {"val_D_Loss": D_loss, "val_G_Loss": G_loss}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        for batch_idx, (x, _) in enumerate(self.test_dl):
            y_fake = self.gen(x)

            if batch_idx == 0:
                y_fake_grid = torchvision.utils.make_grid(
                    y_fake, normalize=True)
                self.logger.experiment.add_image(
                    "fake_test_images", y_fake_grid, global_step=self.current_epoch)

        save_image(y_fake, f"out/epoch_{self.current_epoch}.jpg")

    def configure_optimizers(self):
        opt_disc = optim.Adam(params=self.disc.parameters(),
                              lr=self.lr, betas=self.betas)
        opt_gen = optim.Adam(params=self.gen.parameters(),
                             lr=self.lr, betas=self.betas)

        return opt_disc, opt_gen
