import json
import logging
import os
from glob import glob
from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("medium")


import rootutils

# Setup root directory
root = rootutils.setup_root(
                    search_from=__file__,
                    indicator=[".project-root",'.git'],
                    project_root_env_var=True,             # set the PROJECT_ROOT environment variable to root directory
                    dotenv=True,                           # load environment variables from .env if exists in root directory
                    pythonpath=True,                       # add root directory to the PYTHONPATH (helps with imports)
                    cwd=True                               # change current working directory to the root directory (helps with filepaths)
        )
#----------------------------------------------------------------------------------------


from src.backend.torch_local.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)
from src.backend.torch_local.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)


@task_wrapper
def train(
    cfg: DictConfig,
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
):
    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule)
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
):
    log.info("Starting testing!")
    if trainer.checkpoint_callback.best_model_path:
        log.info(
            f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule=datamodule, verbose=True)
    log.info(f"Test metrics:\n{test_metrics}")
    return test_metrics


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg=cfg))

    pl.seed_everything(seed=3,workers=True,verbose=True)

    # Set up paths
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup('fit')
    datamodule.setup('test')

    imgs,_labels = next(iter(datamodule.train_dataloader()))

    # print(type(tuple(cfg.model.depths)))
    # Initialize Model
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: list[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    if cfg.get("train"):
        train(cfg, trainer, model, datamodule)

    # Test the model
    if cfg.get("test"):
        test_metrics = test(cfg, trainer, model, datamodule)

    if cfg.get('script',False):

        classname_file =  os.path.join(f"{cfg.paths.root_dir}","checkpoints",'classnames',f"{cfg.name}.json")
        if not os.path.isfile(classname_file):
            with open(classname_file,'w') as f:
                json.dump(datamodule.train_ds.class_to_idx,f)


        ptfiles = glob( os.path.join( f"{cfg.paths.root_dir}","checkpoints","pths" ,f"{cfg.name}*.pt" ))
        for ptfile in ptfiles: os.remove(ptfile)

        model_file_path:str = os.path.join( f"{cfg.paths.root_dir}","checkpoints","pths" ,f"{cfg.name}.pt" )
        print(model_file_path)

        imgs,_lbls = next(iter(datamodule.train_dataloader()) )

        # cuda not available: cpu.pt files
        if not torch.cuda.is_available():
            model = model.cpu()
            imgs  = imgs.cpu()
            cpu_model_file_path:str = os.path.join( f"{cfg.paths.root_dir}","checkpoints","pths" ,f"{cfg.name}_cpu.pt" )
            cpu_scripted_model = model.cpu().to_torchscript(method='trace',example_inputs=imgs)
            if os.path.isfile( cpu_model_file_path ):
                os.remove(cpu_model_file_path)
            torch.jit.save(cpu_scripted_model, cpu_model_file_path)
            print(f"torch script model saved: {cpu_model_file_path=}")

        # cuda available: 
        if torch.cuda.is_available():
            # convert model and image into cuda()
            model = model.cuda()
            imgs  = imgs.cuda()

            scripted_model = model.to_torchscript(method='trace',example_inputs=imgs)
            if os.path.isfile( model_file_path ):
                os.remove(model_file_path)

            torch.jit.save(scripted_model, model_file_path)
            print(f"torch script model saved: {model_file_path=}")

            # save file in cpu format also
            model = model.cpu()
            imgs  = imgs.cpu()
            cpu_model_file_path:str = os.path.join( f"{cfg.paths.root_dir}","checkpoints","pths" ,f"{cfg.name}_cpu.pt" )
            cpu_scripted_model = model.cpu().to_torchscript(method='trace',example_inputs=imgs)
            if os.path.isfile( cpu_model_file_path ):
                os.remove(cpu_model_file_path)
            torch.jit.save(cpu_scripted_model, cpu_model_file_path)
            print(f"torch script model saved: {cpu_model_file_path=}")

        onnx_model_file_path:str = os.path.join( f"{cfg.paths.root_dir}","checkpoints","onnxs", f"{cfg.name}.onnx" )
        print(onnx_model_file_path)
        if os.path.isfile( onnx_model_file_path ):
            print("removing old onnx files!!")
            os.remove(onnx_model_file_path)

        model.to_onnx(file_path=onnx_model_file_path, verbose=False, input_sample=imgs, export_params=True, dynamic_axes={'input': {0: 'batch'}},input_names=['input'],output_names=['output'])
        print(f"onnx model saved: {onnx_model_file_path}")

    dataset_name = cfg["data"]["data_dir"].split("/")[-2]
    print(f"Dataset name - {dataset_name}")

    # Dynamic variables
    # name = "vegfruits"
    accuracy = test_metrics[0].get('test/acc_epoch')

    # File path
    filename = f"output_{dataset_name}.txt"

    # Optional: Delete the file if it exists
    if os.path.exists(filename):
        os.remove(filename)

    # Create a new file and write the dynamic content
    with open(filename, "w") as f:
        f.write(f"{dataset_name}_accuracy={accuracy}\n")

    # Return Float for Hparams ::returning train_loss for optuna to compare 'test/acc_epoch','test/loss_epoch'
    # test_metrics =  [ { 'test/loss_epoch':?? , 'test/acc_epoch': ?? } ]
    return test_metrics[0].get('test/loss_epoch')


if __name__ == "__main__":
    main()
