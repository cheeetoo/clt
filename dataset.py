import torch
from torch.utils.data import Dataset
import numpy as np


class CLTActivationDataset(Dataset):
    def __init__(self, bin_path: str, meta_path: str):
        meta = torch.load(meta_path)
        self.shape = meta["shape"]
        self.logical_dtype = meta["logical_dtype"]
        self.storage_dtype = meta["storage_dtype"]

        self.acts = np.memmap(
            bin_path, mode="r", dtype=self.storage_dtype, shape=self.shape
        )

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        raw = torch.from_numpy(self.acts[idx])
        act = raw.view(self.logical_dtype)
        return act[..., 0], act[..., 1]
