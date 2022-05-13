from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_metric
from typing import List

import numpy as np

import torch

from data_loader import load_captions_from_chunks, LabelledTokensDataset, GzippedJSONDataset, Caption

class SponsorTokenClassification:
	def __init__(self, model_path: str = None):
		if model_path:
			self.load(model_path)
		else:
			self.load("bert-base-cased")


	def load(self, model_path):
		# load the tokeniser
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)

		# load the fine-tuned model
		self.model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2)

	def finetune(self,
			dataset_dir: str,
			batch_size: int = 4,
			learning_rate: float = 1e-5,
			weight_decay: float = 0.00001,
			num_train_epochs: int = 5,
			save_path: str = 'SponsorML.model',
			checkpoint_path: str = 'test_sponsors',
			**kwargs
		):

		train_dataset = load_captions_from_chunks('data', chunks=range(1, 12))
		labelled_train_dataset = LabelledTokensDataset(train_dataset, self.tokenizer)

		eval_dataset = GzippedJSONDataset('data.16.json.gz', 100)
		labelled_eval_dataset = LabelledTokensDataset(eval_dataset, self.tokenizer)

		args = TrainingArguments(
			checkpoint_path,
			learning_rate=learning_rate,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			num_train_epochs=num_train_epochs,
			weight_decay=weight_decay,
			**kwargs
		)


		# use SeqEval as the evaluation library
		metric = load_metric("seqeval")

		# define which metrics will be reported
		def compute_metrics(p):
			predictions, labels = p
			predictions = np.argmax(predictions, axis=2)

			# Remove ignored index (special tokens)
			true_predictions = [
				[p for (p, l) in zip(prediction, label) if l != -100]
				for prediction, label in zip(predictions, labels)
			]
			true_labels = [
				[l for (p, l) in zip(prediction, label) if l != -100]
				for prediction, label in zip(predictions, labels)
			]

			results = metric.compute(predictions=true_predictions, references=true_labels)
			return {
				"precision": results["overall_precision"],
				"recall": results["overall_recall"],
				"f1": results["overall_f1"],
				"accuracy": results["overall_accuracy"],
			}

		# specify components of the training and evaluation processes
		trainer = Trainer(
			self.model,
			args,
			train_dataset=labelled_train_dataset,
			eval_dataset=labelled_eval_dataset,
			tokenizer=self.tokenizer,
			compute_metrics=compute_metrics
		)

		trainer.train()
		trainer.evaluate()
		trainer.save_model(save_path)
	
	def predict(self, captions: List[Caption], return_raw_token_labels: bool = False):
		"""
		Predicts the sponsor segments in the given transcript.
		Args:
			captions: transcript of a video in the format of a list of Caption objects.
			return_raw_token_labels (default False): set to true to return the token-level labels
			instead of caption index ranges.
		"""
		token_captions = []
		input_ids = []
		for i, caption in enumerate(captions):
			tokenized_caption = self.tokenizer(caption['text'], add_special_tokens=False)['input_ids']
			input_ids += tokenized_caption
			token_captions += [i] * len(tokenized_caption)

		predicted_labels = []
		for window_start in range(0, len(input_ids), 512):
			if window_start + 512 < len(input_ids):
				w_input_ids = input_ids[window_start:window_start + 512]
			else:
				w_input_ids = input_ids[window_start:]

			with torch.no_grad():
				predictions = self.model.forward(input_ids=torch.tensor(w_input_ids).unsqueeze(0))
				# softmax is applied on the outputs of the previous step
				predictions = list(torch.argmax(predictions.logits.squeeze(), axis=1))

			predicted_labels += predictions

		
		predicted_ranges = []
		in_sponsor = False
		start = -1
		for i, label in enumerate(predicted_labels):
			if label == 1:
				if not in_sponsor:
					in_sponsor = True
					start = i
			elif label == 0:
				if in_sponsor:
					predicted_ranges.append((token_captions[start], token_captions[i-1]))
				in_sponsor = False
		else:
			if in_sponsor:
				predicted_ranges.append((token_captions[start], token_captions[-1]))
		
		return predicted_ranges