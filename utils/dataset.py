import glob
from typing import Callable

import numpy as np
import torch

from .wav2img import *


class WAVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        f_size: int = 512,
        t_size: int = 1024,
        virtual_samplerate: int = 48000,
        use_numpy: bool = True,
        dtype: str = None,
        device: str = None,
        axis: str = "FTC",
        transform: Callable = None,
    ):
        """_summary_

        Args:
            path (str): path to get .wav files
            f_size (int, optional): refers to the resolution of the frequency. Defaults to 512.
            t_size (int, optional): refers to the time length. Defaults to 1024.
            virtual_samplerate (int, optional): Normalize wav file sample rate. Defaults to 48000.
            use_numpy (bool, optional): If True, serve data with numpy array. Otherwise, it is provided as a torch.Tensor . Defaults to True.
            dtype (str, optional): dtype of serving. Defaults to None.
            device (str, optional): only if `use_numpy=False`, configure to where tensor is placed. Defaults to None.
            axis (str, optional): Defines the axis order. F is frequency, T is time, C is channel (Real and Imag). Defaults to "FTC".
            transform (Callable, optional): _description_. Defaults to None.

        Available torch.dtypes:
            https://pytorch.org/docs/stable/tensors.html

            The following are valid cases:
              dtype="float32" # for numpy
              dtype="torch.bfloat16" # for torch
              dtype="torch.float" # for torch

        Available torch device names:
            https://github.com/pytorch/pytorch/blob/7b8cf1f7366bff95e9954037a58a8bb0edaaebd3/c10/core/Device.cpp#L52
        """
        assert not use_numpy or device is None
        assert all(I in axis for I in "FTC")
        assert all(I in "FTC" for I in axis)
        assert len(axis) == len("FTC")

        self.f_size = f_size
        self.t_size = t_size
        self.axis = axis

        data = [
            wav2pfft(fpath, virtual_samplerate, {"nperseg": f_size * 2 - 1})[0]
            for fpath in glob.glob(f"{path}/**.wav")
        ]  # currently data have FTC format
        data = [IMG for IMG in data if IMG.shape[1] >= t_size]

        index = [IMG.shape[1] + 1 - t_size for IMG in data]
        self.length = sum(index)
        self.index = np.cumsum(index, axis=0)

        joined_shape = list(data[0].shape)
        joined_shape[1] = sum(I.shape[1] for I in data)

        axis_order = dict(map(reversed, enumerate("FTC")))
        axis_order = [axis_order[I] for I in axis]
        joined_shape = [joined_shape[I] for I in axis_order]

        if use_numpy:
            self.data = np.empty(joined_shape, dtype=dtype)
        else:
            self.data = torch.empty(joined_shape, dtype=eval(dtype), device=device)

        self.time_axis = axis.find("T")
        time_index = 0
        for DATA in data:
            D = DATA.transpose(axis_order)
            size = D.shape[self.time_axis]

            if not use_numpy:
                D = torch.from_numpy(D)

            if self.time_axis == 0:
                self.data[time_index : time_index + size, :, :] = D
            elif self.time_axis == 1:
                self.data[:, time_index : time_index + size, :] = D
            elif self.time_axis == 2:
                self.data[:, :, time_index : time_index + size] = D

            time_index += size

        if transform is not None:
            self.data = transform(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_id = np.searchsorted(self.index, index, side="right")
        # pixel_id = index - (self.index[img_id - 1] if img_id > 0 else 0)
        pixel_id = index + (self.t_size - 1) * img_id

        if self.time_axis == 0:
            return self.data[pixel_id : pixel_id + self.t_size, :, :]
        elif self.time_axis == 1:
            return self.data[:, pixel_id : pixel_id + self.t_size, :]
        elif self.time_axis == 2:
            return self.data[:, :, pixel_id : pixel_id + self.t_size]


AudioDataset = WAVDataset
