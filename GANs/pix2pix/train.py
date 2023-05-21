from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import codes.config as config
from codes.dataModule import FNAC_seg_pair_dm, MapDataModule
from codes.pixTopix import pixToPix

mdl_ckpt = ModelCheckpoint(
    config.MDL_CKPT_PATH, filename=config.MDL_FILENAME, monitor='G_Loss', mode='min', save_last=True, save_top_k=3
)

erl_stp = EarlyStopping('D_Loss', patience=10, mode='min')

callbacks = [erl_stp, mdl_ckpt]

logger = TensorBoardLogger(
    save_dir=config.LOG_SAVE_DIR, name=config.EXP_NAME, version=config.VERSION)

dm = MapDataModule(root_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE)

model = pixToPix(bs=config.BATCH_SIZE, lr=config.LR)

trainer = Trainer(
    accelerator=config.ACCELERATOR,
    devices=[0],
    precision='16-mixed',
    logger=logger,
    callbacks=callbacks,
    min_epochs=100,
    max_epochs=500,
    log_every_n_steps=10)

if __name__ == '__main__':
    trainer.fit(model=model, datamodule=dm)
    trainer.validate(model=model, datamodule=dm)
