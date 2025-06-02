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
import torch
from torchmetrics import Accuracy


class LitSportModel(pl.LightningModule):
    def __init__(
                self,
                in_chans:int,
                num_classes:int,
                global_pool,    # 'avg',  # 'fast' , 'max' , catavgmax
                depths,         # =(1, 1, 3, 1),
                dims,           # =(24, 48, 22, 168),
                heads,          # =(2, 2, 2, 2),
                global_block_counts, # =(0, 1, 1, 1),
                kernel_sizes,   # =(1, 3, 5, 7),
                d2_scales,      # =(2, 2, 3, 4),
                use_pos_emb,    # =(False, True, False, False),
                ls_init_value=1e-6,
                head_init_scale=1.,
                expand_ratio=4,
                downsample_block=False,     # true
                conv_bias=True,             # false
                stem_type='patch',          # overlap
                head_norm_first=False,
                act_layer=torch.nn.GELU,
                drop_path_rate=0.,
                drop_rate=0.,
                lr = 1e-3,
                weight_decay= 1e-5

        ):
        super().__init__()
        print(self.hparams)
        self.save_hyperparameters()

        # 🛠️ Let's Think and Waste Time  on Hparams 🛠️
        # 🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺🦺

        # self.model = timm.models.edgenext.EdgeNeXt(
        #     in_chans= self.hparams.in_chans,
        #     num_classes= self.hparams.num_classes,
        #     global_block_counts= tuple(self.hparams.global_block_counts),
        #     global_pool= self.hparams.global_pool,
        #     depths = tuple(self.hparams.depths),
        #     dims= tuple(self.hparams.dims),
        #     heads= tuple(self.hparams.heads),
        #     kernel_sizes= tuple(self.hparams.kernel_sizes),
        #     d2_scales= tuple(self.hparams.d2_scales),
        #     use_pos_emb= tuple(self.hparams.use_pos_emb),
        #     ls_init_value=self.hparams.ls_init_value,
        #     head_init_scale=self.hparams.head_init_scale,
        #     expand_ratio=self.hparams.expand_ratio,
        #     downsample_block=self.hparams.downsample_block,
        #     conv_bias=self.hparams.conv_bias,
        #     stem_type=self.hparams.stem_type,
        #     head_norm_first=self.hparams.head_norm_first,
        #     act_layer= torch.nn.GELU if self.hparams.act_layer=='gelu' else torch.nn.ReLU ,
        #     drop_path_rate=self.hparams.drop_path_rate,
        #     drop_rate=self.hparams.drop_rate,
        # )

        # ⌛ FOr Time Being ⌛
        self.model = timm.create_model('edgenext_xx_small.in1k',pretrained=True,num_classes=self.hparams.num_classes)

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
