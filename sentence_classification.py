import os
import sys
from typing import Iterable, List, Optional
import itertools
import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch

# Discover local files as modules
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from log import error, info
from data_loader import load_captions_from_chunks

def get_or_default(arr, i, default):
    return arr[i] if i < len(arr) else default

DATASET_NAME = get_or_default(sys.argv, 1, 'data')
ROOT_DIR = get_or_default(sys.argv, 2, '.')

RE_SEGMENTED = re.compile(r"[.\?\-, ]?,,,", flags=re.MULTILINE)

class TextNormalizer(BaseEstimator, TransformerMixin):

    def remove_whitespace(self, text: str):
        text = text.replace('&nbsp;', ' ')
        return ' '.join((token for token in text.split() if len(token) > 0))

    def normalize_text(self, text: str) -> str:
        text = self.remove_whitespace(text)
        return text

    def normalize(self, document):
        video_id, captions, sponsor_ranges = document

        for caption in captions:
            caption['text'] = self.normalize_text(caption['text'][0])

        return video_id, captions, sponsor_ranges

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document)

class SentenceNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document)


def colored_captions(captions: List[dict], color: str = 'yellow') -> str:
    from termcolor import colored
    return '\n'.join((
        colored(caption['text'], color, attrs=['bold'])
        if caption['is_sponsor'] else caption['text']
        for caption in captions
    ))

# Test that everything works
ds = load_captions_from_chunks(DATASET_NAME, ROOT_DIR)
loader = torch.utils.data.DataLoader(ds, num_workers=0)
normalizer = TextNormalizer()

for document in normalizer.transform(itertools.islice(loader, 50)):
    video_id, captions, sponsor_range = document
    print(video_id, colored_captions(captions))
