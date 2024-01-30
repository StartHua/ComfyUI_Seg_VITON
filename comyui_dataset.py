from os.path import join as opj

import cv2
import numpy as np
from torch.utils.data import Dataset

class Comfyui_Dataset(Dataset):
    def __init__(
            self,
             
            img_fn,
            cloth_fn,
            agn,
            agn_mask,
            cloth,
            image,
            image_densepose,  

            **kwargs
        ):
        self.img_fn = img_fn
        self.cloth_fn = cloth_fn
        self.agn = agn
        self.agn_mask = agn_mask
        self.cloth = cloth
        self.image = image
        self.image_densepose = image_densepose 

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return dict(
            agn=self.agn,
            agn_mask=self.agn_mask,
            cloth=self.cloth,
            image=self.image,
            image_densepose=self.image_densepose,
            txt="",
            img_fn=self.img_fn,
            cloth_fn=self.cloth_fn,
        )