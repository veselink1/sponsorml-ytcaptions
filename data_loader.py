from typing import Generator, Iterable, List, Optional, Tuple
from glob import glob
import os
import random

from numpy import isin
import pandas as pd
import torch

from log import info, warn

# Captions are often coded to stay on screen until the next caption line comes
# but that is not ideal -- so we limit how much it is reasonable for a word
# to stay on screen. The issue with leaving captions on for longer is that in some
# edge cases, the captions are incorrectly left unchanged throughout a sponsored
# segment (the sponsored segment is not transcribed) and when it comes to
# labelling the captions as part of our process to construct a training dataset,
# these "left-over" captions from before the sponsored segment starts are the
# the only text coded to be on-screen during the sponsored segment.
# I.e. this is done to aim in data cleaning.
#
# For slow English speakers, each words takes about 0.54 seconds to pronounce.
# We are limit each token to 1 second.
# Source: https://debatrix.com/en/tools/speech-calculator/#:~:text=How%20many%20words%20per%20minute,will%20use%20around%20110%20words.
MAX_DURATION_PER_TOKEN = 1

# Classes for the sequence classification model
SPONSOR_CLASS = 'sponsor'
CONTENT_CLASS = 'content'

class Caption(dict):
	"""
	A caption line with start and end time in seconds.
	"""
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

def get_intersection_range(captions: List[dict], start: float, end: float, error: float = 1) -> Tuple[int, int]:
	"""
	Intersects the list of captions with the start and end times of a segment expressed
	in seconds. Allows for an error margin for captions to be considered part of the segment.
	The error margin is also in seconds.

	Returns `(None, None)` if no such sub-list exists.
	"""
	start_caption = None
	for i in range(len(captions)):
		if captions[i]['start'] >= start:
			start_caption = i
			break

	if start_caption is None:
		return None, None

	# Apply the error margin
	if start_caption + 1 < len(captions):
		# If the second caption starts very close to the start of the segment,
		# skip over the first caption that was picked originally; it is likely
		# non related to the sponsored content.
		if captions[start_caption + 1]['start'] - start < error:
			start_caption += 1

	end_caption = None
	for i in range(start_caption, len(captions)):
		if captions[i]['start'] >= end:
			end_caption = i
			break

	if end_caption is None:
		return None, None

	# Apply the error margin
	if end_caption - 1 >= 0:
		# If the second to last caption ends very close to the end of the
		# segment, skip over the last caption that was picked originally.
		# It is likely non related to the sponsored content.
		if captions[end_caption - 1]['end'] - end < error:
			end_caption = max(start_caption, end_caption - 1)

	assert start_caption <= end_caption
	return start_caption, end_caption

def tokenize(text: str) -> List[str]:
	"""
	Tokenizes the string.
	"""
	# Split by unicode whitespace.
	return text.split()

def tokenize_caption_list(captions: Iterable[Caption]):
	"""
	Tokenizes the list of captions by splitting captions with multiple tokens
	into separate `Caption` objects. The duration for each token is approximated
	from the duration of the original caption and the length of the work in characters.
	This allows for more accurate token-level extraction to be done.

	Input: 0:0:0 Hello World 0:0:2
	Output: 0:0:0 Hello 0:0:1 World 0:0:2
	"""
	for caption in captions:
		text = caption.text
		tokens = tokenize(text)
		if len(tokens) == 0:
			continue
		if len(tokens) == 1:
			yield Caption(caption.start, caption.end, tokens[0])
			continue

		# Heuristic: Assume the time necessary to pronouce each word depends
		# on the number of characters in that word.
		total_chars = sum(len(token) + 1 for token in tokens)
		time_per_char = (caption.end - caption.start) / total_chars

		current_timestamp = caption.start
		for token in tokens:
			token_duration = min(time_per_char * len(token), MAX_DURATION_PER_TOKEN)
			yield Caption(current_timestamp, current_timestamp + time_per_char * len(token), token)
			current_timestamp += token_duration + time_per_char

def segment_duration(captions: List[Caption]):
	"""
	Returns the duration of the segment in seconds.
	"""
	if len(captions) < 1:
		return 0
	return captions[-1].end - captions[0].start

def segment_text(captions: List[Caption]):
	"""
	Returns the text in the segment.
	"""
	return ' '.join((caption.text for caption in captions))

def forward_time_window(captions: List[Caption], start_index: int, duration: int):
	"""
	Seeks forward for `duration` seconds starting at the start time of the
	caption with index `start_index` the captions are fully contained within
	that time interval.
	"""
	results = [captions[start_index]]
	start_time = captions[start_index].start
	index = start_index + 1
	while index < len(captions) and captions[index].end < start_time + duration:
		results.append(captions[index])
		index += 1

	return results

def backward_time_window(captions: List[Caption], end_index: int, duration: int):
	"""
	Seeks backward for `duration` seconds ending at the start time of the caption
	with index `end_index` and returns the captions are fully contained within
	that time interval.
	"""
	results = [captions[end_index]]
	end_time = captions[end_index].end
	index = end_index - 1
	while index >= 0 and captions[index].start > end_time - duration:
		results.insert(0, captions[index])
		index -= 1

	return results

def extract_labelled_data(video_id, captions, sponsor_ranges):
	"""
	Extracts labelled examples suitable for the training of a binary
	sequence classification model.

	For each video 2*N examples are generated, where N is the number of sponsored
	segments in the video.

	The text of each sponsored segment is matched to other random text in the video
	occuring in a similar time frame (if the sponsored segment is 10s,
	the non-sponsored segment extracted is also 10s).
	"""
	sponsor_segments = []
	content_segments = []

	# Segment the whole video into sponsored and content segments.
	last_sponsor_segment_end = -1
	for start, end in sponsor_ranges:
		sponsor_segments.append(captions[start:end])
		content_segments.append(captions[last_sponsor_segment_end + 1:start])
		last_sponsor_segment_end = end

	# Don't forget the last segment (could be empty, but we check for that later).
	content_segments.append(captions[last_sponsor_segment_end + 1:])

	# Shuffle the content segments before matching sponsor segments to them
	random.shuffle(content_segments)

	for sponsor_segment in sponsor_segments:
		if len(sponsor_segment) == 0:
			continue

		sponsor_duration = segment_duration(sponsor_segment)

		# Match to a non-sponsor segment with the same duration
		found_match = False

		for content_segment in content_segments:
			if segment_duration(content_segment) < sponsor_duration:
				continue

			# Sample either the first N or last N seconds of the segment
			if bool(random.getrandbits(1)):
				short_content_segment = forward_time_window(content_segment, 0, sponsor_duration)
			else:
				short_content_segment = backward_time_window(content_segment, len(content_segment) - 1, sponsor_duration)

			# Yield the two segments
			yield segment_text(sponsor_segment), SPONSOR_CLASS
			yield segment_text(short_content_segment), CONTENT_CLASS

			# Process the next pair
			found_match = True
			break

		if not found_match:
			warn(f'Could not find a non-sponsored segment with the same duration as the sponsored segment for {video_id}')


class GzippedJSONDataset(torch.utils.data.IterableDataset):
	"""
	Reads a .json.gz file.
	"""
	def __init__(self, path: str, subset_length: int = None):
		super().__init__()
		self.path = path
		self.subset_length = subset_length

	def __iter__(self) -> Generator[(str, List[dict], List[Tuple[int, int]])]:
		info(f'Opening {self.path} for reading...')
		count = 0
		with pd.read_json(self.path, orient='record', lines=True, compression='infer', chunksize=500) as reader:
			for chunk in reader:
				for video_id, captions, sponsor_times in chunk.itertuples(index=False, name=None):
					if self.subset_length and self.subset_length < count:
						break
					count += 1
					# Convert from tuples to dicts
					captions = [Caption(start, end, text) for (text, start, end) in captions]
					yield video_id, captions, sponsor_times
		info(f'Closed {self.path}.')

class LabelledCaptionsDataset(torch.utils.data.IterableDataset):
	"""
	An `IterableDataset` which labels and returns the captions from the source
	dataset. Suitable for the training of a sequence labelling model.

	Yields tuples of the form: `video_id, captions, sponsor_ranges`.
	The list of captions is a list of `Caption` with an extra key `is_sponsor`
	indicating whether the caption belongs to a sponsored segment in the video.
	"""

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

			if len(sponsor_ranges) == 0:
				drop_row = True

			if not drop_row:
				yield video_id, captions, sponsor_ranges

class LabelledExamplesDataset(torch.utils.data.IterableDataset):
	"""
	An `IterableDataset` which samples the video transcripts and yields
	examples of the classes `sponsor` or `content`. Due to how the sampling is
	done, the dataset is always balanced in terms of class labels and sequence
	lengths from both classes.

	Yields tuples of the form: `text, label`.

	See also: `extract_labelled_data`
	"""

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

class LabelledTokensDataset(torch.utils.data.IterableDataset):
	"""
	IterableDataset that tokenizes the transcripts of a given caption dataset
	and labels the tokens according to whether they are included in a sponsor-labeled caption.
	"""

	def __init__(self, dataset, tokenizer):
		super().__init__()
		self.dataset = dataset
		self.tokenizer = tokenizer

	def __iter__(self):
		for video_id, captions, sponsor_times in self.dataset:

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
				input_ids = []
				labels = []

				for caption in captions:
					tokenized_caption = self.tokenizer(caption['text'])
					# remove special beginning/end tokens
					input_ids += tokenized_caption['input_ids'][1:-1]

					# label every token accordingly
					label = 1 if 'is_sponsor' in caption else 0
					labels += [label] * len(tokenized_caption['input_ids'][1:-1])

				# flag indicating whether a completely non-sponsor segment has been yielded.
				# limitting the number of fully non-sponsor segments to balance the data
				yielded_non_sponsor = False

				# go through the transcript max_length segment by max_length segment
				for window_start in range(0, len(input_ids), 510):
					w_input_ids = input_ids[window_start:]
					w_labels = labels[window_start:]

					# make sure to yield at most 1 completely non-sponsor segment
					if 1 not in w_labels:
						if yielded_non_sponsor:
							continue
						else:
							yielded_non_sponsor = True

					# add back special tokens
					prepared_tokenizer = self.tokenizer.prepare_for_model(w_input_ids, truncation=True, padding='max_length')

					attention_mask = prepared_tokenizer['attention_mask']

					# loop to deal with special tokens, labelling them with -100 for the BERT model to ignore
					new_labels = []
					for i, m in enumerate(attention_mask):
						if m == 1:
							if i == 0:
								new_labels.append(-100)
							elif i == len(attention_mask) - 1:
								new_labels.append(-100)
							elif i == len(w_input_ids) + 1:
								new_labels.append(-100)
							else:
								new_labels.append(w_labels[i-1])
						else:
							new_labels.append(-100)

					w_input_ids = prepared_tokenizer['input_ids']

					yield {'input_ids': w_input_ids, 'labels': new_labels, 'attention_mask': attention_mask}

	def __len__(self):
		# needed for training, 20001 is the length of every chunk in the caption dataset
		return 20001


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

	return torch.utils.data.ChainDataset([GzippedJSONDataset(file) for file in files])

def load_captions_from_chunks(base_name: str, root_dir: str = '.', chunks: Optional[Iterable[int]] = None):
	"""
	Loads all `data.N.csv.gz` files and labels the individual captions.

	See also: `LabelledCaptionsDataset`
	"""

	return LabelledCaptionsDataset(load_data_from_chunks(base_name, root_dir, chunks))

def load_examples_from_chunks(base_name: str, root_dir: str = '.', chunks: Optional[Iterable[int]] = None):
	"""
	Loads all `data.N.csv.gz` files and produces labelled examples.

	See also: `LabelledExamplesDataset`
	"""

	return LabelledExamplesDataset(load_data_from_chunks(base_name, root_dir, chunks))
