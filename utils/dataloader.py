import glob

import numpy as np
import torch

from .wav2img import *


class WAVDataset(torch.utils.data.Dataset):
    def __init__(self, root, dtype=np.uint8, image_size=None, virtual_samplerate=48000):
        self.image_size = image_size
        nperseg = image_size * 2 - 1
        self.data = [
            wav2img(fpath, dtype, {"nperseg": nperseg}, virtual_samplerate)[0]
            for fpath in glob.glob(f"{root}/**.wav")
        ]
        index = [IMG.shape[1] + 1 - image_size for IMG in self.data]
        self.length = sum(index)
        self.index = np.cumsum(index, axis=0)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_id = np.searchsorted(self.index, index, side="right")
        pixel_id = index - (self.index[img_id - 1] if img_id > 0 else 0)
        img = self.data[img_id][:, pixel_id : pixel_id + self.image_size]
        return img
