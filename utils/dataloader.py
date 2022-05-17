import glob

import numpy as np
import torch

from .wav2img import *


class WAVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        dtype: np.dtype = np.uint8,
        image_size: int = 512,
        virtual_samplerate: int = 48000,
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
            data_config (_type_, optional): Configuration of how data is stored and returned. Defaults to { "use_numpy": True, "dtype": "uint8", "device": None, "axis": -1, }.

        To return in torch **bfloat16** tensor format and preload on gpu:
            data_config = {
                "use_numpy": False,
                "dtype": "torch.BFloat16Tensor",
                "device": "cuda",
                "axis": -1,
            }

        information about multiple dtypes:
            https://pytorch.org/docs/stable/tensors.html
        """
        assert not data_config["use_numpy"] or data_config["device"] is None

        self.data_config = data_config
        self.image_size = image_size
        nperseg = image_size * 2 - 1
        self.data = [
            wav2img(fpath, dtype, {"nperseg": nperseg}, virtual_samplerate)[0]
            for fpath in glob.glob(f"{path}/**.wav")
        ]
        index = [IMG.shape[1] + 1 - image_size for IMG in self.data]
        self.length = sum(index)
        self.index = np.cumsum(index, axis=0)

        axis = [1, 0, 2]
        if data_config["axis"] not in [-1, 2]:
            axis = axis[: data_config["axis"]] + [2] + axis[data_config["axis"] : -1]
        self.data = [x.transpose(*axis) for x in self.data]
        self.joined_axis = [i for i, I in enumerate(axis) if I == 1][0]
        self.swap_axis = [0, 1, 2]
        self.swap_axis.pop(data_config["axis"])
        self.data = np.concatenate(self.data, axis=self.joined_axis)

        if data_config["use_numpy"]:
            self.data = self.data.astype(data_config["dtype"])
        else:
            self.data = torch.from_numpy(self.data)
            self.data = self.data.type(data_config["dtype"])

        if data_config["device"] is not None:
            self.data = self.data.to(data_config["device"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_id = np.searchsorted(self.index, index, side="right")
        # pixel_id = index - (self.index[img_id - 1] if img_id > 0 else 0)
        pixel_id = index + (self.image_size - 1) * img_id

        if self.joined_axis == 1:
            img = self.data[:, pixel_id : pixel_id + self.image_size]
        elif self.joined_axis == 0:
            img = self.data[pixel_id : pixel_id + self.image_size]

        if self.data_config["use_numpy"]:
            img = img.swapaxes(*self.swap_axis)
        else:
            img = img.transpose(*self.swap_axis)
        return img


AudioDataset = WAVDataset
