import os
import sys
from collections import defaultdict
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.realpath('..')))
from data_loader import Caption, segment_text, get_intersection_range

os.environ["WANDB_DISABLED"] = "true"

def caption_times(c):
	return c.start, c.end

def prediction_times(p):
	return tuple(p[0])

def tumbling_time_window(captions, duration, key=caption_times):
	results = [captions[0]]
	for caption in captions:
		if key(results[-1])[1] - key(results[0])[0] <= duration:
			results.append(caption)
		else:
			yield results
			results = [caption]

	yield results

def session_time_window(captions, duration, key=caption_times):
	captions_iter = iter(captions)
	results = [next(captions_iter)]
	for caption in captions_iter:
		if key(results[-1])[1] - key(caption)[0] <= duration:
			results.append(caption)
		else:
			yield results
			results = [caption]

	yield results

def batch(iterable, n):
	length = len(iterable)
	for i in range(0, length, n):
		yield iterable[i:min(i + n, length)]

def decode_label(outputs):
	content, sponsor = outputs

	prediction_dict = {'sponsor': sponsor, 'content': content}
	prediction_dict = {k: v for k, v in sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True)}

	return next(iter(prediction_dict.items()))

def merge_prediction_(predictions):
	assert len(set((label for _, _, label, _ in predictions))) == 1
	# All co-occurring predictions have the same label so we merge them
	merged_start, merged_end = predictions[0][0][0], predictions[-1][0][1]
	merged_text = ' '.join((text for _, text, _, _ in predictions))
	# Don't know what the correct way to compute the joint probability here is,
	# just assume they are independent; We don't really use this number anywhere
	prob = np.prod([prob for _, _, _, prob in predictions])
	return [merged_start, merged_end], merged_text, predictions[0][2], prob

def merge_predictions(predictions, within_duration=5):
	for co_occuring in session_time_window(predictions, within_duration, key=prediction_times):
		merged = [co_occuring[0]]
		for times, text, label, prob in co_occuring[1:]:
			_, _, prev_label, _ = merged[0]
			if label == prev_label:
				merged.append((times, text, label, prob))
			else:
				yield merge_prediction_(merged)
				merged = [(times, text, label, prob)]

		if len(merged) > 0:
			yield merge_prediction_(merged)

class SponsorSequenceClassification:
	def __init__(self, model_path: str = None, window_duration: int = 10, range_prob_threshold: float = 0.9, device: str = 'cuda'):
		self.model_path = model_path
		self.window_duration = window_duration
		self.range_prob_threshold = range_prob_threshold
		self.device = device

		if model_path is None:
			self.load('distilbert-base-uncased')
		else:
			self.load(model_path)


	def load(self, model_path):
		# load the tokeniser
		self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

		# load the fine-tuned model
		self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
		self.model.to(self.device)

	def tokenize(self, examples):
		return self.tokenizer(examples["text"], padding="max_length", truncation=True)

	def predict(self, captions: List[Caption]):
		predicted_times = []
		predicted_probs = []
		for times, text, label, prob in merge_predictions(self.predict_sponsor_segments_(captions, self.window_duration), self.window_duration):
			if label == 'sponsor' and prob > self.range_prob_threshold:
				predicted_times.append(times)
				predicted_probs.append(prob)

		predicted_ranges = [get_intersection_range(captions, *pair) for pair in predicted_times]
		return predicted_ranges

	def predict_in_batches_(self, texts, batch_size: int = 8):
		batches = list(batch(texts, batch_size))
		for b in batches:
			inputs = defaultdict(list)
			for text in b:
				tokenized = self.tokenize({ 'text': text })
				for k, v in tokenized.items():
					inputs[k].append(v)

			inputs = { k: torch.tensor(v).to(self.device) for k, v in inputs.items() }
			outputs = self.model(**inputs)
			predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()
			yield from predictions

	def predict_sponsor_segments_(self, captions, window_duration=10):
		windows = list(tumbling_time_window(captions, window_duration))
		window_texts = [segment_text(window) for window in windows]
		predictions = self.predict_in_batches_(window_texts, 4)

		for window, text, prediction in zip(windows, window_texts, predictions):
			yield [window[0].start, window[-1].end], text, *decode_label(prediction)
