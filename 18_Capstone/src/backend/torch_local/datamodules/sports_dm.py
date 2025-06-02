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

import os
from pathlib import Path
from typing import AnyStr

import lightning as pl
import torchvision
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import DatasetFolder
from torchvision import transforms

from src.backend.torch_local.utils.helpers import custom_check_image, custom_loader

__all__ = ['LitSportsDataModule']

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class LitSportsDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers:int ,
            pin_memory:bool,
            data_dir:AnyStr | None=None
        ):
        print("calling Sports  🤾‍♂️ DataModule")
        super().__init__()
        for _ in ['train','test','valid']:
            assert _ in os.listdir(data_dir), f"expected dirs missing <{_}>"
        self.batch_size:int  = batch_size
        self.num_workers:int = num_workers
        self.pin_memory:bool = pin_memory
        self.data_dir:AnyStr = data_dir
        self.save_hyperparameters(ignore=['data_dir'])

    def prepare_data(self):
        self.train_dir: Path = os.path.join(self.data_dir,'train')
        self.test_dir:  Path = os.path.join(self.data_dir,'test')
        self.valid_dir: Path = os.path.join(self.data_dir,'valid')

    def setup(self, stage=None):
        if stage is None:
            raise ValueError('nothing loaded in the DataModule!!')
        if stage=='fit':
            self.train_ds:DatasetFolder  = DatasetFolder(root=self.train_dir,loader=custom_loader,is_valid_file=custom_check_image,transform=img_transform)
            self.valid_ds:DatasetFolder  = DatasetFolder(root=self.valid_dir,loader=custom_loader,is_valid_file=custom_check_image,transform=img_transform)
        elif stage=='test':
            self.test_ds:DatasetFolder   = DatasetFolder(root=self.test_dir,loader=custom_loader,is_valid_file=custom_check_image,transform=img_transform)   #,transform=Compose([Resize((224,224)),ToTensor])

    def train_dataloader(self):
        return DataLoader(self.train_ds,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,collate_fn=default_collate,pin_memory=self.pin_memory) #,generator=torch.Generator(device=device)

    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,collate_fn=default_collate,pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_ds,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,collate_fn=default_collate,pin_memory=self.pin_memory)
