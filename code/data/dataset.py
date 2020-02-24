import os
import pandas as pd
import numpy as np
import torch
import multiresolutionimageinterface as mri


class DatasetWSI(torch.utils.data.Dataset):
    """
    Returns a sample of the data
    """
    def __init__(self, dir, file_names):
        self.dir = dir
        self.file_names = file_names

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        return file_name

    def __len__(self):
        return len(self.file_names)
    
    