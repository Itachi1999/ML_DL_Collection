from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from gen_model import Generator
from disc_model import Discriminator


class pixToPix(pl.LightningModule):
    def __init__(
        self, bs=8, lr=2e-4, in_channels=3, l1_lambda=100, lambda_gp=10, betas=(0.5, 0.999),
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

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
        if self.global_step % 10 == 0:
            x_grid = torchvision.utils.make_grid(x, nrow=2, normalize=True)
            y_grid = torchvision.utils.make_grid(y, nrow=2, normalize=True)
            y_fake_grid = torchvision.utils.make_grid(
                y_fake, nrow=2, normalize=True)

            self.experiment.add_image(
                "FNAC_real_images", x_grid, global_step=self.global_step)
            self.experiment.add_image(
                "FNAC_segmentation_masks", y_grid, global_step=self.global_step)
            self.experiment.add_image(
                "FNAC_fake_images", y_fake_grid, global_step=self.global_step)

        log_dict = {"D_Loss": D_loss, "G_Loss": G_loss}
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        opt_disc = optim.Adam(params=self.disc.parameters(),
                              lr=self.lr, betas=self.betas)
        opt_gen = optim.Adam(params=self.gen.parameters(),
                             lr=self.lr, betas=self.betas)

        return opt_disc, opt_gen
