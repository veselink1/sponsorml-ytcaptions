import os
import pandas as pd

class Caption:
	def __init__(self, start: int, end: int, text: str):
		self.start = start
		self.end = end
		self.text = text

	def __repr__(self):
		return self.text

def get_caption_list_from_path(path: str) -> list[Caption]:
	captions = []
	with open(path) as f:
		lines = f.readlines()

		# remove double \n then remove \n
		lines = "".join(lines).replace('\n\n\n', '\n\n').split('\n')

		# where to split
		# first section is header and last one is an extra linebreak
		breakpoints = [i for i, line in enumerate(lines) if line == ''][1:-1]

		start = breakpoints[0] + 1
		for end in breakpoints[1:]:
			timestamps, text = lines[start], " ".join(lines[start+1:end])
			text = clean_text(text)
			timestamps = timestamps.split(' ')
			timestamps = [timestamps[0], timestamps[2]]

			start_time, end_time = [_get_timestamp_in_seconds(ts) for ts in timestamps]

			captions.append(Caption(start_time, end_time, text))
			start = end + 1

	return captions

def _get_timestamp_in_seconds(timestamp: str) -> float:
	h, m, s = [float(x) for x in timestamp.split(':')]
	return (h * 3600) + (m * 60) + s

# allow an error margin for the caption to be considered part of the segment
def get_intersection_range(captions: list[Caption], start: float, end: float, error: float = 0.2) -> (float, float):
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

def prepare_data(captions_path: str, sponsorml_path: str, output_path: str = 'data.csv', vote_threshold: int = 20):
	filenames = os.listdir(captions_path)
	sponsorml_df = pd.read_csv(sponsorml_path)
	rows = []

	for filename in filenames:
		videoID = filename.split('.')[0]
		# get all labeled segments of video
		segments = sponsorml_df[sponsorml_df.videoID == videoID]
		for _, segment in segments.iterrows():
			# only check sponsor segments and those with votes >= input threshold
			if segment.category != 'sponsor' or segment.votes < vote_threshold:
				continue

			captions = get_caption_list_from_path(f'{captions_path}/{filename}')

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
	df.to_csv('data.csv')


CAPTIONS_PATH = 'captions'
SPONSORML_PATH = 'sponsorTimes.csv'
OUTPUT_PATH = 'data.csv'
VOTE_THRESHOLD = 20

def main():
	prepare_data(CAPTIONS_PATH, SPONSORML_PATH, OUTPUT_PATH, VOTE_THRESHOLD)
	
if __name__ == '__main__':
	main()
