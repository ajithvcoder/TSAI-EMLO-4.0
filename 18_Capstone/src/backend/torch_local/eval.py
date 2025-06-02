import logging
from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("medium")

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


# Imports that require root directory setup
import os

from src.backend.torch_local.utils.helpers import distribution_fn, show_batch_images
from src.backend.torch_local.utils.logging_utils import (
    plot_confusion_matrix,
    setup_logger,
    task_wrapper,
)

log = logging.getLogger(__name__)


def instantiate_callbacks(callback_cfg: DictConfig) -> list[pl.Callback]:
    callbacks: list[pl.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    loggers: list[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
):
    log.info("Starting testing!")
    if cfg.ckpt_path:
        log.info(
            f"Loading best checkpoint: {cfg.ckpt_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=cfg.ckpt_path
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
    log.info(f"Test metrics:\n{test_metrics}")
    return test_metrics


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg=cfg))

    # Set up paths
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup('fit')  # required for confusion matrix
    datamodule.setup('test')

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # model.load_from_checkpoint(cfg.ckpt_path)

    # Set up callbacks
    callbacks: list[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers,
        callbacks=callbacks
    )

    # # Test the model
    if cfg.get("test"):
        _test_metrics = test(cfg, trainer, model, datamodule)

    distribution_fn(datamodule.train_dataloader(),path=os.path.join(cfg.paths.root_dir,'assets'))

    images, _labels  = next(iter(datamodule.train_dataloader()))
    show_batch_images(images,path=os.path.join(cfg.paths.root_dir,'assets'))

    plot_confusion_matrix(model=model,datamodule=datamodule,classes=datamodule.test_ds.classes,path=os.path.join(cfg.paths.root_dir,'assets'))


if __name__ == "__main__":
    main()
