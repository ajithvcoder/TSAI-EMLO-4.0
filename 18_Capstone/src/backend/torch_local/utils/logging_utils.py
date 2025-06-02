import os
import sys
from functools import wraps

import lightning as pl
import rootutils
import torch
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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


def setup_logger(log_file):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(log_file, rotation="10 MB")


def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise

    return wrapper


def get_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


def plot_confusion_matrix(
    model: pl.LightningModule, datamodule: pl.LightningDataModule,classes:list, path: str = "."
):
    model.eval()
    os.makedirs(path, exist_ok=True)

    def fn(mode: str, loader: torch.utils.data.DataLoader):
        y_pred = []
        y_true = []
        for batch in loader():
            x, y = batch
            logits = model(x)
            _loss = torch.nn.functional.cross_entropy(logits, y)
            preds = torch.nn.functional.softmax(logits, dim=-1)
            # preds,true comes in batch(32)
            preds = torch.argmax(preds, dim=-1)
            for i, j in zip(preds, y, strict=False):
                # print(y.shape,preds.shape,type(y),type(preds))
                y_true.append(j.item())
                y_pred.append(i.item())

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print(cm)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes
        )
        disp.plot(xticks_rotation="vertical", colorbar=False).figure_.savefig(
            os.path.join(path, f"{mode}_confusion_matrix.png")
        )  # f'{path}{mode}_confusion_matrix.png')

    for mode, loader in zip(
        ["val","train", "test"],
        [
            datamodule.train_dataloader,
            datamodule.test_dataloader,
            datamodule.val_dataloader,
        ], strict=False,
    ):
        fn(mode=mode, loader=loader)
