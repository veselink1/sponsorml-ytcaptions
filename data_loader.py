from typing import Generator, Iterable, List, Optional, Tuple
from glob import glob
import os
import gzip
import random

from numpy import isin
import pandas as pd
import torch

from log import info, warn

class Caption(dict):
    def __init__(self, start: float, end: float, text: str):
        self['text'] = text
        self['start'] = start
        self['end'] = end

    @property
    def text(self):
        return self['text']

    @text.setter
    def text(self, value):
        self['text'] = value

    @property
    def start(self):
        return self['start']

    @start.setter
    def start(self, value):
        self['start'] = value

    @property
    def end(self):
        return self['end']

    @end.setter
    def end(self, value):
        self['end'] = value


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


# captions are often coded to stay on screen until the next caption line comes
# but that is not ideal -- so we limit how much it is reasonable for a word
# to stay on screen. Source: https://debatrix.com/en/tools/speech-calculator/#:~:text=How%20many%20words%20per%20minute,will%20use%20around%20110%20words.
# For slow english speakers, each words takes about 0.54 seconds to pronounce.
MAX_DURATION_PER_TOKEN = 1
CAPTION_TIME_WINDOW = 5
SPONSOR_BEGIN_CLASS = 'sponsor_begin'
SPONSOR_END_CLASS = 'sponsor_end'
NON_SPONSOR_CLASS = 'not_sponsor'

def tokenize(text: str) -> List[str]:
	# basic tokenization
	return text.split()

def tokenize_caption_list(captions: Iterable[Caption]):
	for caption in captions:
		text = caption.text
		tokens = tokenize(text)
		if len(tokens) == 0:
			continue
		if len(tokens) == 1:
			yield Caption(caption.start, caption.end, tokens[0])
			continue

		# assume every character is pronounced and
		# characters and word separators take the same amount of time
		total_chars = sum(len(token) + 1 for token in tokens)
		time_per_char = (caption.end - caption.start) / total_chars

		current_timestamp = caption.start
		for token in tokens:
			token_duration = min(time_per_char * len(token), MAX_DURATION_PER_TOKEN)
			yield Caption(current_timestamp, current_timestamp + time_per_char * len(token), token)
			current_timestamp += token_duration + time_per_char

def forward_time_window(captions: List[Caption], start_index: int, duration: int):
	results = [captions[start_index]]
	start_time = captions[start_index].start
	index = start_index + 1
	while index < len(captions) and captions[index].end < start_time + duration:
		results.append(captions[index])
		index += 1

	return results

def backward_time_window(captions: List[Caption], end_index: int, duration: int):
	results = [captions[end_index]]
	end_time = captions[end_index].end
	index = end_index - 1
	while index >= 0 and captions[index].start > end_time - duration:
		results.insert(0, captions[index])
		index -= 1

	return results

def segment_duration(captions: List[Caption]):
	if len(captions) < 1:
		return 0
	return captions[-1].end - captions[0].start

def segment_text(captions: List[Caption]):
	return ' '.join((caption.text for caption in captions))

def extract_labelled_data(video_id, captions, sponsor_ranges):
	sponsor_segments = []
	non_sponsor_segments = []
	last_sponsor_segment_end = -1
	for start, end in sponsor_ranges:
		sponsor_segments.append(captions[start:end])
		non_sponsor_segments.append(captions[last_sponsor_segment_end + 1:start])
		last_sponsor_segment_end = end

	non_sponsor_segments.append(captions[last_sponsor_segment_end + 1:])

	# Sort by segment length
	random.shuffle(sponsor_segments)
	random.shuffle(non_sponsor_segments)

	for sponsor_segment in sponsor_segments:
		if segment_duration(sponsor_segment) < CAPTION_TIME_WINDOW * 2:
			print(f'Skipping short sponsor segment: {segment_duration(sponsor_segment)}s')
			continue

		sponsor_begin_segment = forward_time_window(sponsor_segment, 0, CAPTION_TIME_WINDOW)
		sponsor_end_segment = backward_time_window(sponsor_segment, len(sponsor_segment) - 1, CAPTION_TIME_WINDOW)

		if len(sponsor_begin_segment) * MAX_DURATION_PER_TOKEN < CAPTION_TIME_WINDOW:
			print(f'Skipping sponsor segment with weird captioning: {sponsor_begin_segment}')
			continue
		if len(sponsor_end_segment) * MAX_DURATION_PER_TOKEN < CAPTION_TIME_WINDOW:
			print(f'Skipping sponsor segment with weird captioning: {sponsor_end_segment}')

		sponsor_begin_text = segment_text(sponsor_begin_segment)
		sponsor_end_text = segment_text(sponsor_end_segment)

		if sponsor_begin_text == sponsor_end_text:
			print('Skipping sponsor segment with broken captions')
			continue

		# Match to a non-sponsor segment with the length
		found_match = False

		for non_sponsor_segment in non_sponsor_segments:
			if segment_duration(non_sponsor_segment) <= CAPTION_TIME_WINDOW * 3:
				continue

			# Remove from list so we don't pick it again
			non_sponsor_segments.remove(non_sponsor_segment)

			mid_point = len(forward_time_window(non_sponsor_segment, 0, CAPTION_TIME_WINDOW))
			if mid_point >= len(non_sponsor_segment):
				# Skip
				continue

			short_non_sponsor_segment = forward_time_window(non_sponsor_segment, mid_point, CAPTION_TIME_WINDOW)
			non_sponsor_text = segment_text(short_non_sponsor_segment)

			# Yield the three segments
			yield sponsor_begin_text, SPONSOR_BEGIN_CLASS
			yield sponsor_end_text, SPONSOR_END_CLASS
			yield non_sponsor_text, NON_SPONSOR_CLASS

			# Process the next pair
			found_match = True
			break

		if not found_match:
			print(f'Could not find a non-sponsored segment with the same length as the sponsored segment for {video_id}')


class GzippedJSONDataset(torch.utils.data.IterableDataset):
    """
    Reads a .json.gz file.
    """
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self) -> Generator[(str, List[dict], List[Tuple[int, int]])]:
        info(f'Opening {self.path} for reading...')
        with pd.read_json('./data.1.json.gz', orient='record', lines=True, compression='infer', chunksize=500) as reader:
            for chunk in reader:
                for video_id, captions, sponsor_times in chunk.itertuples(index=False, name=None):
                    # Convert from tuples to dicts
                    captions = [Caption(start, end, text) for (text, start, end) in captions]
                    yield video_id, captions, sponsor_times
        info(f'Closed {self.path}.')

class LabelledCaptionsDataset(torch.utils.data.IterableDataset):

    def __init__(self, dataset: torch.utils.data.IterableDataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for video_id, captions, sponsor_times in self.dataset:
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

class LabelledExamplesDataset(torch.utils.data.IterableDataset):

    def __init__(self, dataset: torch.utils.data.IterableDataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for video_id, captions, sponsor_times in self.dataset:
            captions = list(tokenize_caption_list(captions))
            segment_ranges = [get_intersection_range(captions, start_time - MAX_DURATION_PER_TOKEN, end_time + MAX_DURATION_PER_TOKEN, error=MAX_DURATION_PER_TOKEN) for start_time, end_time in sponsor_times]
            # Filter out broken ranges
            segment_ranges = [r for r in segment_ranges if r[0] is not None and r[1] is not None]

            for text, label in extract_labelled_data(video_id, captions, segment_ranges):
                if len(text) == 0:
                    continue
                yield text, label

def load_data_from_chunks(base_name: str, root_dir: str = '.', chunks: Optional[Iterable[int]] = None):
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

    return torch.utils.data.ChainDataset([GzippedJSONDataset(file) for file in files])

def load_captions_from_chunks(base_name: str, root_dir: str = '.', chunks: Optional[Iterable[int]] = None):
    """
    Loads all `data.N.csv.gz` files and labels the individual captions.
    """

    return LabelledCaptionsDataset(load_data_from_chunks(base_name, root_dir, chunks))

def load_examples_from_chunks(base_name: str, root_dir: str = '.', chunks: Optional[Iterable[int]] = None):
    """
    Loads all `data.N.csv.gz` files and produces labelled examples.
    """

    return LabelledExamplesDataset(load_data_from_chunks(base_name, root_dir, chunks))
