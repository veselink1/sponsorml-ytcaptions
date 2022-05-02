from typing import Generator, Iterable, List, Optional, Tuple
from glob import glob
import os
import gzip
from numpy import isin

import pandas as pd
import torch

from log import info, warn

def parse_int_tuple(s):
    return tuple(map(int, s.replace('(','').replace(')','').split(',')))

# allow an error margin for the caption to be considered part of the segment
def get_intersection_range(captions: List[dict], start: float, end: float, error: float = 1) -> Tuple[int, int]:
    start_caption = None
    for i in range(len(captions)):
        if captions[i]['start'] >= start:
            start_caption = i
            break

    if start_caption is None:
        return None, None

    if start_caption + 1 < len(captions):
        if captions[start_caption + 1]['start'] - start < error:
            start_caption += 1

    end_caption = None
    for i in range(start_caption, len(captions)):
        if captions[i]['start'] >= end:
            end_caption = i
            break

    if end_caption is None:
        return None, None

    if end_caption - 1 >= 0:
        if captions[end_caption - 1]['end'] - end < error:
            end_caption = max(start_caption, end_caption - 1)

    assert start_caption <= end_caption
    return start_caption, end_caption


class GzippedCSVDataset(torch.utils.data.IterableDataset):
    """
    Reads a .csv.gz file.
    """
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self) -> Generator[(str, List[dict], List[Tuple[int, int]])]:
        info(f'Opening {self.path} for reading...')
        for row in pd.read_json(self.path, orient='record', compression='infer').itertuples(index=False, name=None):

            video_id, captions, sponsor_times = row
            for caption in captions:
                caption['is_sponsor'] = False

            drop_row = False

            sponsor_ranges = []

            for start_time, end_time in sponsor_times:
                # get intersection range and extract the sponsor text from it
                start_index, end_index = get_intersection_range(captions, start_time, end_time)
                if start_index is None or end_index is None:
                    print(f'Dropping {video_id} because sponsor times do not match the captions')
                    drop_row = True
                    break

                # mark range as sponsor
                for i in range(start_index, end_index):
                    captions[i]['is_sponsor'] = True

                sponsor_ranges.append([start_index, end_index])

            if not drop_row:
                yield video_id, captions, sponsor_ranges
        info(f'Closed {self.path}.')

def load_captions_from_chunks(base_name: str, root_dir: str = '.', chunks: Optional[Iterable[int]] = None):
    """
    Loads all `data.N.csv.gz` files.
    """
    if chunks is None:
        files = glob(os.path.join(root_dir, f'{base_name}.*.json.gz'))
    else:
        files = [os.path.join(root_dir, f'{base_name}.{chunk}.json.gz') for chunk in chunks]

    if len(files) == 0:
        raise Exception('No matching files found!')

    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        info(f'Found {file}.')

    return torch.utils.data.ChainDataset([GzippedCSVDataset(file) for file in files])
