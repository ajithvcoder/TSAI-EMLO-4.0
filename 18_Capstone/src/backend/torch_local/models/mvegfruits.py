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
#-----------------------------------------------------------------------------------------------------------------------------------------

import lightning as pl
import timm
import ast
import os
import torch
from torchmetrics import Accuracy


class LitVegFruitsModel(pl.LightningModule):
    def __init__(
            self,
            base_model: str,
            # dims: List,
            # depths: List,
            # head_fn: str,
            # conv_ratio: float,
            patch_size: int,
            embed_dim: int,
            num_classes: int,
            lr: float,
            weight_decay: float,
            pretrained=False,
            # trainable: bool,
            in_chans: int = 3,
            **kwargs

        ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.save_hyperparameters()

        # self.model:timm.models.mambaout.MambaOut = timm.models.mambaout.MambaOut(
        #     in_chans=self.hparams.in_chans,
        #     num_classes=self.hparams.num_classes,
        #     depths=list(self.hparams.depths),
        #     dims=list(self.hparams.dims),
        #     head_fn=self.hparams.head_fn,
        #     conv_ratio=self.hparams.conv_ratio,
        #     act_layer=torch.nn.ReLU,
        # )

        self.model = timm.create_model(base_model,pretrained=pretrained,num_classes=self.hparams.num_classes,
                    dims=ast.literal_eval(kwargs['dims']), depths=ast.literal_eval(kwargs['depths']))

        self.train_acc: Accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.test_acc: Accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.valid_acc: Accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=-1)
        self.train_acc(preds, y)
        self.log("train/loss", loss.item(), prog_bar=True, on_epoch=True, on_step=True)
        self.log(
            "train/acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=True
        )
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=-1)
        self.test_acc(preds, y)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("test/acc", self.test_acc, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=-1)
        self.valid_acc(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("val/acc", self.valid_acc, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #                 optimizer=optimizer,
        #                 factor=self.hparams.scheduler_factor,
        #                 patience=self.hparams.scheduler_patience,
        #                 min_lr=self.hparams.min_lr
        # )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams.lr * 10,
            three_phase=True,
            final_div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            # "monitor":"train/loss_epoch"
        }
