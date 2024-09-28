import os
from pathlib import Path
import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dogbreed_modules import DogBreedImageDataModule
from models.dogbreed_classifer import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper

@task_wrapper
def train_and_save(data_module, model, trainer, save_path="../model_storage/model.ckpt"):
    trainer.fit(model, data_module)
    # trainer.test(model, data_module)
    # model.save_model(save_path)
    # print(f"Model saved to {save_path}")

class CustomModelCheckpiont(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        trainer.lightning_module.save_transformed_model = True
        super()._save_checkpoint(trainer, filepath)
        # print(filepath)

def main(args):
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / args.data
    log_dir = base_dir / args.logs
    ckpt_path = base_dir / args.ckpt_path
    
    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    data_module = DogBreedImageDataModule(dl_path=data_dir, batch_size=32, num_workers=0)

    # Initialize Model
    model = DogBreedClassifier(lr=1e-3)

    # Set up checkpoint callback
    checkpoint_callback = CustomModelCheckpiont(
        dirpath=ckpt_path,
        filename="{epoch}-checkpoint",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=1,
        callbacks=[
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ],
        accelerator="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name="dogbreed_classification"),
    )

    # Train and test the model
    train_and_save(data_module, model, trainer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer using trained Dogbreed Classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to data containing images")
    parser.add_argument("--logs", type=str, required=True, help="Path to logs")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    main(args)