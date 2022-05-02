import os
import sys
from time import time
from typing import Iterator, List, Optional, Tuple
from enum import Enum
from datetime import timedelta

import pandas as pd
import webvtt # webvtt-py

class Caption:
	def __init__(self, start: int, end: int, text: str):
		self.start = start
		self.end = end
		self.text = text
		# self.is_sponsor = False

	def __repr__(self):
		text = self.text.replace('\n', '\\n')
		return f'Caption{{{self.start}:{self.end},{text}}}'

class DBSegment:
	category: str
	startTime: float
	endTime: float
	UUID: str
	userID: str
	votes: int
	views: int
	locked: int
	hidden: int
	shadowHidden: int
	videoID: str
	videoDuration: int
	reputation: int
	hashedVideoID: str
	timeSubmitted: int
	userAgent: str
	service: str
	description: str

class OverlappingSegmentGroup:
	def __init__(self):
		self.segments: List[DBSegment] = []
		self.votes = 0

def get_caption_list_from_path(path: str):
	cap_iter: Iterator[webvtt.Caption] = webvtt.read(path)
	# The cap_iter may contain overlapping caption regions like below:
	#	<Caption start=00:00:00.060 end=00:00:01.730 text= \nyou know working with tech all the time>
	#	<Caption start=00:00:01.730 end=00:00:01.740 text=you know working with tech all the time\n >
	#	<Caption start=00:00:01.740 end=00:00:03.470 text=you know working with tech all the time\nI sometimes forget that not everyone>
	#	<Caption start=00:00:03.470 end=00:00:03.480 text=I sometimes forget that not everyone\n >
	#	<Caption start=00:00:03.480 end=00:00:05.059 text=I sometimes forget that not everyone\nknows about even the most basic of>
	#	<Caption start=00:00:05.059 end=00:00:05.069 text=knows about even the most basic of\n >
	#	<Caption start=00:00:05.069 end=00:00:07.579 text=knows about even the most basic of\nthings example well the various>
	#	<Caption start=00:00:07.579 end=00:00:07.589 text=things example well the various\n >
	#	<Caption start=00:00:07.589 end=00:00:10.070 text=things example well the various\nstandards of network cables cat 5 versus>
	#	<Caption start=00:00:10.070 end=00:00:10.080 text=standards of network cables cat 5 versus\n >
	# This is how auto-generated captions follow a person's speech.
	# cap[1] replaces cap[0] on-screen and creates a sliding effect.
	# This is fine for YT but not for us.

	# To fix remove the duplication, we check for overlapping parts of text
	# between the captions and eliminate them.
	output: List[Caption] = []
	prev_cap = Caption(start=0, end=0, text="")
	for cap in cap_iter:
		cap = Caption(cap.start_in_seconds, cap.end_in_seconds, cap.text)
		cap.text = cap.text.strip()

		if cap.text == '':
			continue

		ilen = get_intersection_length(prev_cap.text, cap.text)
		# Is the overlap a whole token or more?
		if ilen >= len(cap.text.split(' ', 1)[0]):
			if len(cap.text) == ilen:
				# Remove the whole caption altogether, it is duplicated
				continue
			else:
				# Remove the overlap from the previous caption
				prev_cap.text = prev_cap.text[:-ilen]

		output.append(cap)
		prev_cap = cap

	return output

def get_intersection_length(left: str, right: str):
	"""
	Finds how many characters of overlap is there between left and right.

	```
	left  = "except the various"
	right =            "various forms"
	                    ^^^^^^^
	```
	"""
	i = 0
	while not right.startswith(left[i:]):
		i += 1
	ilen = len(left) - i
	assert ilen == 0 or left[-ilen:] == right[:ilen]
	return ilen

def tokenize(text: str) -> List[str]:
	# basic tokenization
	return text.split(' ')

def clean_text(text: str) -> str:
	# no cleaning yet
	return text

def build_segment_groups(segments: List[DBSegment]) -> List[OverlappingSegmentGroup]:
	"""
	This function will find segments that are contained inside of eachother, called similar segments.
	Segments with less than -1 votes are already ignored before this function is called.

	Based on https://github.com/ajayyy/SponsorBlockServer/blob/e74b985304443b17b429c5c82696c7a03e78a166/src/routes/getSkipSegments.ts#L276
	"""

	# Create groups of segments that are similar to eachother
	# Segments must be sorted by their startTime so that we can build groups chronologically:
	# 1. As long as the segments' startTime fall inside the currentGroup, we keep adding them to that group
	# 2. If a segment starts after the end of the currentGroup (> cursor), no other segment will ever fall
	#    inside that group (because they're sorted) so we can create a new one
	overlappingSegmentsGroups: List[OverlappingSegmentGroup] = []
	currentGroup = None
	cursor = -1 # -1 to make sure that, even if the 1st segment starts at 0, a new group is created
	for segment in segments:
		if segment.startTime >= cursor:
			currentGroup = OverlappingSegmentGroup()
			overlappingSegmentsGroups.append(currentGroup)

		currentGroup.segments.append(segment)
		# only if it is a positive vote, otherwise it is probably just a sponsor time with slightly wrong time
		if segment.votes > 0:
			currentGroup.votes += segment.votes

		cursor = max(cursor, segment.endTime)

	return overlappingSegmentsGroups

def get_best_segment(group: OverlappingSegmentGroup):
	"""
	SponsorBlock chooses a segment from an overlap group randomly by using the
	votes property as a weight. This is done so that all segments can have a
	chance of appearing and makes sense in that system, but here we just
	want the best possible match, hence we pick the segment with the highest vote.
	"""

	return max(group.segments, key=lambda segment: segment.votes)

def prepare_data(captions_path: str, sponsorml_path: str, output_path: str, vote_threshold: int, chunk_size: int):
	filenames = os.listdir(captions_path)
	print('Reading database...')
	sponsorml_df = pd.read_csv(sponsorml_path)
	rows = []

	start_time = time()

	chunk_id = 0
	def write_chunk():
		nonlocal chunk_id
		nonlocal rows
		chunk_id += 1
		df = pd.DataFrame(rows, columns=['videoID', 'captions', 'sponsor_times'])
		filename, ext = output_path.rsplit('.json', 1)
		chunk_filename = f'{filename}.{chunk_id}.json{ext}'
		df.to_json(chunk_filename, orient='records', compression='infer')
		rows = []

	# This is much much faster than doing sponsorml_df[sponsorml_df.videoID == videoID]
	# later. (cut down running time from 12h to 3h)
	grouped_df = sponsorml_df.groupby(by=["videoID"])

	print('Processing captions...')
	for i, filename in enumerate(filenames):
		progress = (i + 1) / len(filenames) * 100
		elapsed = time() - start_time
		remaining = (100 - progress) * elapsed / progress
		print('\u001b[2K\r', end='')
		print(f'{progress:.2f}% ' +
			f'elapsed: {timedelta(seconds=int(elapsed))}, ' +
			f'remaining: {timedelta(seconds=int(remaining))}',
			end='', flush=True)
		videoID = filename.split('.')[0]
		# get all labeled segments of video

		segments = grouped_df.get_group(videoID)

		if len(segments) == 0:
			print(f'No segments for {videoID}!')

		# segment filtering
		# https://github.com/ajayyy/SponsorBlockServer/blob/e74b985304443b17b429c5c82696c7a03e78a166/src/routes/getSkipSegments.ts#L18
		segments = [segment for _, segment in segments.iterrows()
			if segment.category == 'sponsor' and segment.votes >= vote_threshold] # TODO: Think about adding a threshold on `views`
		# Filter out similar segments
		segments = [get_best_segment(group) for group in build_segment_groups(segments)]

		try:
			captions = get_caption_list_from_path(f'{captions_path}/{filename}')
		except Exception as e:
			e.args = (*e.args, f'while processing {filename}')
			raise e

		segment_times = [(segment.startTime, segment.endTime) for segment in segments]

		rows.append((videoID, captions, segment_times))

		if len(rows) > chunk_size:
			write_chunk()

	if len(rows) > 0:
		write_chunk()

def get_or_default(arr, i, default):
	return arr[i] if i < len(arr) else default

CAPTIONS_PATH = get_or_default(sys.argv, 1, 'captions')
SPONSORML_PATH = get_or_default(sys.argv, 2, 'sponsorTimes.csv')
OUTPUT_PATH = get_or_default(sys.argv, 3, 'data.json.gz')
VOTE_THRESHOLD = -1 # Same threshold SponsorBlock is using
CHUNK_SIZE = 10_000

def main():
	prepare_data(CAPTIONS_PATH, SPONSORML_PATH, OUTPUT_PATH, VOTE_THRESHOLD, CHUNK_SIZE)

if __name__ == '__main__':
	main()
