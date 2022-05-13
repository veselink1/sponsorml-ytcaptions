import os
import sys
from collections import defaultdict
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

sys.path.append(os.path.dirname(os.path.realpath('..')))
from data_loader import Caption, segment_text, get_intersection_range

os.environ["WANDB_DISABLED"] = "true"

def index_of_token(offset_mapping, char_index, default_value):
    for i, r in enumerate(offset_mapping):
        if i == 0:
            # Skip the [CLS]
            continue
        if r[0] <= char_index <= r[1]:
            return i

    return default_value

class SponsorSpanExtraction:
	def __init__(self, model_path: str = None, device: str = 'cuda'):
		self.model_path = model_path
		self.device = device

		if model_path is None:
			self.load('distilbert-base-uncased')
		else:
			self.load(model_path)


	def load(self, model_path):
		# load the tokeniser
		self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

		# load the fine-tuned model
		self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
		self.model.to(self.device)

	def tokenize(self, examples):
		return self.tokenizer(
			examples['text'],
			max_length=512,
			truncation=True,
			return_offsets_mapping=True,
			padding='max_length',
			stride=128,
		)

	def predict(self, captions: List[Caption]):
		raise NotImplemented()
		# inputs = self.tokenize({ 'text': segment_text(captions) })
		# print(inputs)

		# start_position = index_of_token(inputs['offset_mapping'], start_char_idx, default_value=0)
		# end_position = index_of_token(inputs['offset_mapping'], end_char_idx, default_value=0)
		# if start_position != 0 and end_position == 0:
		# 	end_position = len(inputs['input_ids']) - 2

		# outputs = trained(input_ids=torch.tensor([inputs['input_ids']]).cuda())
