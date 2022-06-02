import glob
from typing import Callable

import numpy as np
import torch

from .config import config as SST_config
from .wav2img import *


class WAVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        config: SST_config,
        transform: Callable = None,
    ):
        """_summary_

        Args:
            path (str): path to get .wav files
            config (SST_config): Details of data processing
            transform (Callable, optional): Additional preprocessing of data. Defaults to None.
        """
        
        self.t_size = t_size = config.t_size
        self.time_axis = config.axis.find("T")
        data = [
            wav2pfft(fpath, config)
            for fpath in glob.glob(f"{path}/**.wav")
        ]
        data = [IMG for IMG in data if IMG.shape[self.time_axis] >= t_size]

        index = [IMG.shape[self.time_axis] + 1 - t_size for IMG in data]
        self.length = sum(index)
        self.index = np.cumsum(index, axis=0)

        joined_shape = list(data[0].shape)
        joined_shape[self.time_axis] = sum(I.shape[self.time_axis] for I in data)

        if config.use_numpy:
            self.data = np.empty(joined_shape, dtype=config.serve_dtype)
        else:
            self.data = torch.empty(joined_shape, dtype=eval(config.serve_dtype), device=config.device)

        time_index = 0
        for DATA in data:
            size = DATA.shape[self.time_axis]

            if not config.use_numpy:
                DATA = torch.from_numpy(DATA)

            if self.time_axis == 0:
                self.data[time_index : time_index + size, :, :] = DATA
            elif self.time_axis == 1:
                self.data[:, time_index : time_index + size, :] = DATA
            elif self.time_axis == 2:
                self.data[:, :, time_index : time_index + size] = DATA

            time_index += size

        if transform is not None:
            self.data = transform(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_id = np.searchsorted(self.index, index, side="right")
        pixel_id = index + (self.t_size - 1) * img_id

        if self.time_axis == 0:
            return self.data[pixel_id : pixel_id + self.t_size, :, :]
        elif self.time_axis == 1:
            return self.data[:, pixel_id : pixel_id + self.t_size, :]
        elif self.time_axis == 2:
            return self.data[:, :, pixel_id : pixel_id + self.t_size]


AudioDataset = WAVDataset
