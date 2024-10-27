from pathlib import Path
from typing import Union, Tuple, List
import os

import shutil
import lightning as L
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

class CustomInferImageFolder(ImageFolder):
    def __init__(self, root, filenames, transform=None):
        super().__init__(root, transform=transform)
        # Filter the dataset based on the provided filenames
        # self.imgs = [os.path.basename(file) for file in filenames]
        self.infer_imgs = [img for img in self.imgs if os.path.basename(img[0]) in filenames]
        # print("self.imgs")
        # print(self.infer_imgs)
        self.imgs = self.infer_imgs
        # print("length-", self.imgs)
        # self.length = 5

    def __getitem__(self, index):
        
        # print("self.infer_imgs-", self.infer_imgs[index])
        path, target = self.imgs[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path
    
    def __len__(self):
        return len(self.imgs)



class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        except Exception as e:
            print(f"Skipping corrupted or unreadable file: {path}")
            return self.__getitem__((index + 1) % len(self.samples))  # Skip to the next image
        return sample, target

class CatDogImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dl_path: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        pin_memory: bool = False,
        samples: int = 10, filenames: List = [], classes: dict = {}
    ):
        super().__init__()
        self._data_dir = Path(dl_path)
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._splits = splits
        self._pin_memory = pin_memory
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._infer_dataset = None
        self._filenames = filenames

    def prepare_data(self):
        pass

    @property
    def data_path(self):
        return self._data_dir

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def create_dataset(self, root, transform):
        return CustomImageFolder(root=root, transform=transform)

    def create_infer_dataset(self, root, filenames, transform):
        return CustomInferImageFolder(root=root, filenames=filenames, transform=transform)

    def reduce_data_debug(self, original_subset, reduce_number):
        original_indices = original_subset.indices
        new_indices = original_indices[:reduce_number]
        new_subset = Subset(original_subset.dataset, new_indices)
        return new_subset

    def setup(self, stage: str = None):
        if self._train_dataset is None:
            train_data = self.create_dataset(
                self.data_path / "train",
                self.train_transform,
            )
            train_size = int(self._splits[0] * len(train_data))
            val_size = len(train_data) - train_size
            self._train_dataset, self._val_dataset = random_split(
                train_data, [train_size, val_size]
            )
            # Todo remove after debugging
            # print(type(self._train_dataset))
            # print(self._train_dataset)
            # self._train_dataset, self._val_dataset = self.reduce_data_debug(self._train_dataset, 500), \
            #         self.reduce_data_debug(self._val_dataset, 50)

        if self._test_dataset is None:
            self._test_dataset = self.create_dataset(
                self.data_path / "test",
                self.valid_transform,
            )

        # TODO: this method loads all the files and then checks for respective files. It causes more processing power
        if self._infer_dataset is None:
            directory = "infer_images"

            # Check if the directory exists
            if os.path.exists(directory):
                # Clear all the data inside the directory
                shutil.rmtree(directory)
                print(f"Directory '{directory}' cleared.")
                
            # Create the directory
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
            self._infer_dataset = self.create_infer_dataset(self.data_path / "test",
                        self._filenames,
                        self.valid_transform)
            # self._batch_size =  1


    def __dataloader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=shuffle,
            pin_memory=self._pin_memory,
        )

    def train_dataloader(self):
        return self.__dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self._val_dataset)

    def test_dataloader(self):
        return self.__dataloader(self._test_dataset)
    
    def predict_dataloader(self):
        self._batch_size =  1
        return self.__dataloader(self._infer_dataset)  # Using validation dataset for testing