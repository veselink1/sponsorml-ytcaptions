import boto3
import datetime
import os
import tarfile
import io
import json
import torch
import pickle
import yt_dlp
from torch.serialization import _load, _open_zipfile_reader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from functools import partial
from glob import glob
from .sequence_labelling import SponsorTokenClassification
from .preprocessing import get_caption_list_from_path
from .dataset_scripts.main import download_en_captions
from yt_dlp import DownloadError, SameFileError

s3 = boto3.client('s3')
os.makedirs('/tmp/caps', exist_ok=True)

def load_model_from_s3(model_path: str, s3_bucket: str, file_prefix: str):
	if model_path and s3_bucket and file_prefix:
		model = SponsorTokenClassification()
		os.makedirs(model_path, exist_ok=True)
		obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
		bytestream = io.BytesIO(obj['Body'].read())
		tar = tarfile.open(fileobj=bytestream, mode="r:gz")
		for member in tar.getmembers():
			if member.name.endswith("/pytorch_model.bin"):
				f = tar.extractfile(member)
				state_dict = torch.load(io.BytesIO(f.read()), map_location=torch.device('cpu'))
			else:
				tar.extract(member, model_path)

		config = AutoConfig.from_pretrained(f'{model_path}/seq_labelling.model/config.json')
		model.tokenizer = AutoTokenizer.from_pretrained(f'{model_path}/seq_labelling.model')
		model.model = AutoModelForTokenClassification.from_pretrained(
					pretrained_model_name_or_path=None, state_dict=state_dict, config=config)

		return model
	else:
		raise KeyError('No S3 Bucket and Key Prefix provided')

model = load_model_from_s3('/tmp/seq_labelling', 'sponsor-ml', 'model/seq_labelling_model.tar.gz')



def get_sponsor_timestamps(video_id: str):
	# captions = download_en_captions('lEaPQ7YrWQg', 'caps', return_captions=True)
	captions = download_en_captions(video_id, '/tmp/caps', return_captions=True)
	# transcript = [c.text for c in captions['captions']]
	# returns indicies of segment
	segments_idx = model.predict(captions['captions'])
	
	# ranges in seconds
	segment_seconds = []

	for s, e in segments_idx: 
		if e - s < 3:
			continue
		# print(s, e)
		# print(" ".join(transcript[s:e]))
		# print()
		start_sec, end_sec = captions['captions'][s].start, captions['captions'][e].end
		segment_seconds.append((start_sec, end_sec))
		# ts_start, ts_end = (str(datetime.timedelta(seconds=seconds)) for seconds in [start_sec, end_sec])

	return segment_seconds

def lambda_handler(event, context):
	video_id = event['queryStringParameters']['video_id']
	return {
		'statusCode': 200,
		'body': get_sponsor_timestamps(video_id)
	}
