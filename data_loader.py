from typing import Iterable, Optional
from glob import glob
import os
import gzip

import pandas as pd
import torch

from log import info

class GzippedCSVDataset(torch.utils.data.IterableDataset):
    """
    Reads a .csv.gz file.
    """
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        info(f'Opening {self.path} for reading...')
        with gzip.open(self.path, 'rb') as reader:
            for row in pd.read_csv(reader).itertuples():
                yield row
        info(f'Closed {self.path}.')

def load_captions_from_chunks(base_name: str, root_dir: str = '.', chunks: Optional[Iterable[int]] = None):
    """
    Loads all `data.N.csv.gz` files.
    """
    if chunks is None:
        files = glob(os.path.join(root_dir, f'{base_name}.*.csv.gz'))
    else:
        files = [os.path.join(root_dir, f'{base_name}.{chunk}.csv.gz') for chunk in chunks]

    if len(files) == 0:
        raise Exception('No matching files found!')

    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        info(f'Found {file}.')

    return torch.utils.data.ChainDataset([GzippedCSVDataset(file) for file in files])
