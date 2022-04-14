import os
import sys
from typing import List, Tuple
import pandas as pd
from enum import Enum

import webvtt # webvtt-py

class Caption:
	def __init__(self, start: int, end: int, text: str):
		self.start = start
		self.end = end
		self.text = text

	def __repr__(self):
		return self.text

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

def get_caption_list_from_path(path: str) -> list[Caption]:
	return list(webvtt.read(path))

def _get_timestamp_in_seconds(timestamp: str) -> float:
	h, m, s = [float(x) for x in timestamp.split(':')]
	return (h * 3600) + (m * 60) + s

# allow an error margin for the caption to be considered part of the segment
def get_intersection_range(captions: list[Caption], start: float, end: float, error: float = 0.2) -> Tuple[float, float]:
	segment_range = [0, 0]
	for i in range(len(captions)):
		if (captions[i].start + error) >= start:
			segment_range[0] = i
			for j in range(i, len(captions)):
				if (captions[j].start - error) >= end:
					segment_range[1] = j
					break
			break
	return tuple(segment_range)

def tokenize(text: str) -> list[str]:
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

def prepare_data(captions_path: str, sponsorml_path: str, output_path: str, vote_threshold: int):
	filenames = os.listdir(captions_path)
	print('Reading database...')
	sponsorml_df = pd.read_csv(sponsorml_path)
	rows = []

	print('Processing captions...')
	for i, filename in enumerate(filenames):
		progress = (i + 1) / len(filenames) * 100
		print(f'\r{progress:.2f}%', end='', flush=True)
		videoID = filename.split('.')[0]
		# get all labeled segments of video
		segments = sponsorml_df[sponsorml_df.videoID == videoID]

		if len(segments) == 0:
			print(f'No segments for {videoID}!')

		# segment filtering
		# https://github.com/ajayyy/SponsorBlockServer/blob/e74b985304443b17b429c5c82696c7a03e78a166/src/routes/getSkipSegments.ts#L18
		segments = [segment for _, segment in segments.iterrows()
			if segment.category == 'sponsor' and segment.votes >= vote_threshold] # TODO: Think about adding a threshold on `views`
		# Filter out similar segments
		segments = [get_best_segment(group) for group in build_segment_groups(segments)]

		for segment in segments:
			try:
				captions = get_caption_list_from_path(f'{captions_path}/{filename}')
			except Exception as e:
				e.args = (*e.args, f'while processing {filename}')
				raise e

			# extract transcript
			full_transcript = " ".join([caption.text for caption in captions])

			# get intersection range and extract the sponsor text from it
			start_index, end_index = get_intersection_range(captions, segment.startTime, segment.endTime)
			sponsor_text = " ".join([caption.text for caption in captions[start_index:end_index]])

			# get the indicies of the first token and last token in the segment
			token_start = 0
			for i in range(start_index):
				token_start += len(tokenize(captions[i].text))

			token_end = token_start + 1
			for i in range(start_index, end_index):
				token_end += len(tokenize(captions[i].text))

			rows.append((videoID, full_transcript, sponsor_text, (token_start, token_end)))

	df = pd.DataFrame(rows, columns=['videoID', 'transcript', 'sponsorText', 'sponsorTokenRange'])
	df.to_csv(output_path)

def get_or_default(arr, i, default):
	return arr[i] if i < len(arr) else default

CAPTIONS_PATH = get_or_default(sys.argv, 1, 'captions')
SPONSORML_PATH = get_or_default(sys.argv, 2, 'sponsorTimes.csv')
OUTPUT_PATH = get_or_default(sys.argv, 3, 'data.csv')
VOTE_THRESHOLD = -1 # Same threshold SponsorBlock is using

def main():
	prepare_data(CAPTIONS_PATH, SPONSORML_PATH, OUTPUT_PATH, VOTE_THRESHOLD)

if __name__ == '__main__':
	main()
