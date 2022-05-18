import glob

import numpy as np
import torch
from typing import Callable

from .wav2img import *


class WAVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        dtype: np.dtype = np.uint8,
        image_size: int = 512,
        virtual_samplerate: int = 48000,
        transform: Callable = None,
        data_config: dict = {
            "use_numpy": True,
            "dtype": "uint8",
            "device": None,
            "axis": -1,
        },
    ):
        """_summary_

        Args:
            path (str): path to get .wav files
            dtype (np.dtype, optional): dtype of array of sound. Defaults to np.uint8.
            image_size (int, optional): image size of sound. Defaults to 512.
            virtual_samplerate (int, optional): Normalize wav file sample rate. Defaults to 48000.
            transform (Callable, optional): A function for change value of data. Defaults to None.
            data_config (_type_, optional): Configuration of how data is stored and returned. Defaults to { "use_numpy": True, "dtype": "uint8", "device": None, "axis": -1, }.

        To return in torch **bfloat16** tensor format and preload on gpu:
            data_config = {
                "use_numpy": False,
                "dtype": "torch.BFloat16Tensor",
                "device": "cuda",
                "axis": -1,
            }

        Available dtypes:
            https://pytorch.org/docs/stable/tensors.html

        Available device names:
            https://github.com/pytorch/pytorch/blob/7b8cf1f7366bff95e9954037a58a8bb0edaaebd3/c10/core/Device.cpp#L52
        """
        assert not data_config["use_numpy"] or data_config["device"] is None

        self.image_size = image_size
        nperseg = image_size * 2 - 1
        self.data = [
            wav2img(fpath, dtype, {"nperseg": nperseg}, virtual_samplerate)[0]
            for fpath in glob.glob(f"{path}/**.wav")
        ]
        index = [IMG.shape[1] + 1 - image_size for IMG in self.data]
        self.length = sum(index)
        self.index = np.cumsum(index, axis=0)

        if data_config["axis"] not in [-1, 2]:
            axis = list(range(3))
            axis = axis[: data_config["axis"]] + [2] + axis[data_config["axis"] : -1]
            self.data = [x.transpose(*axis) for x in self.data]
            self.joined_axis = 2
        else:
            self.joined_axis = 1
        self.data = np.concatenate(self.data, axis=self.joined_axis)

        if data_config["use_numpy"]:
            self.data = self.data.astype(data_config["dtype"])
        else:
            self.data = torch.from_numpy(self.data)
            self.data = self.data.type(data_config["dtype"])

        if data_config["device"] is not None:
            self.data = self.data.to(data_config["device"])
        
        if transform is not None:
            self.data = transform(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_id = np.searchsorted(self.index, index, side="right")
        # pixel_id = index - (self.index[img_id - 1] if img_id > 0 else 0)
        pixel_id = index + (self.image_size - 1) * img_id

        if self.joined_axis == 2:
            img = self.data[:, :, pixel_id : pixel_id + self.image_size]
        elif self.joined_axis == 1:
            img = self.data[:, pixel_id : pixel_id + self.image_size]
        return img


AudioDataset = WAVDataset
