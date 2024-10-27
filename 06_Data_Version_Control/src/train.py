import os
from pathlib import Path
import logging
import comet_ml

import hydra
import torch
import lightning as L
from lightning.pytorch.loggers import Logger 
from typing import List
from omegaconf import DictConfig
from dotenv import load_dotenv
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)
load_dotenv("../.env")

class CustomModelCheckpiont(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        trainer.lightning_module.save_transformed_model = True
        super()._save_checkpoint(trainer, filepath)
        # print(filepath)


def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks
    
    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    # print("callbacks-", callbacks)
    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers
    
    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

@task_wrapper
def train_task(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting training!")
    trainer.fit(model, datamodule)
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    if cfg.get("train"):
        train_task(cfg, trainer, model, datamodule)

if __name__ == "__main__":
    train()