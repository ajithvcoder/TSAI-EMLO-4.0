import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def custom_check_image(path:str)->bool:
  '''
    utility to check file is Image or not
  '''
  try:
    os.path.isfile(path=path)
    _ = Image.open(path)
    return True
  except Exception:
    return False


def custom_loader(path:str,*,size:tuple=(224,224))->NDArray:
    '''
        helper utils to load  image_path to np.array shape:: ( C, H, W )
    '''
    # Image.open  >> ( H , W , C )   >>  ( C, H, W )
    # return np.array(Image.open(path).convert("RGB").resize(size=size))#.transpose(0,1,2) #,dtype=np.float32
    return Image.open(path).convert("RGB").resize(size)


def distribution_fn(dataloader:DataLoader,path=".")->None:
  plt.clf()
  class_distribution:dict = {}
  for batch in dataloader:
      for _class in batch[1]:
          if _class.item() not in class_distribution:
             class_distribution[_class.item()]=1
          else:
            class_distribution[_class.item()]+=1
  ax = pd.DataFrame(index=class_distribution.keys(),data=class_distribution.values()).reset_index().rename(columns={0:'distribution'}).sort_values(by='index').plot(kind='bar',y='distribution',x='index',figsize=(24,6),grid=True,xlabel="no. of classes")
  fig = ax.get_figure()
  fig.savefig(os.path.join(path,'class_distribution.png'))


def show_batch_images(batch:Tensor,path=".")->None:
  '''
    helper utility to show images in a batch
  '''
  assert batch.shape[1::] == (3,224,224),"batch is not proper shape"
  plt.clf()
  plt.imshow(make_grid(batch).permute(1,2,0),aspect='auto')
  plt.axis('off')  # Hide axis
  plt.savefig(os.path.join(path,'image_grid.png'), bbox_inches='tight', pad_inches=0)
