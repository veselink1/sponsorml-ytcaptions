import os
import sys
from typing import Iterable, Optional
import itertools
from log import info

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch

# Discover local files as modules
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from data_loader import load_captions_from_chunks

def get_or_default(arr, i, default):
	return arr[i] if i < len(arr) else default

DATASET_NAME = get_or_default(sys.argv, 1, 'data')
ROOT_DIR = get_or_default(sys.argv, 2, '.')

# Test that everything works
ds = load_captions_from_chunks(DATASET_NAME, ROOT_DIR)
loader = torch.utils.data.DataLoader(ds, num_workers=0)
print(list(itertools.islice(loader, 5)))

# TODO: Preprocess, write the classifier
