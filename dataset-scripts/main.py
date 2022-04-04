from concurrent.futures import thread
from glob import glob
import os
import sys
import multiprocessing.dummy
import subprocess
from sys import stdout
import threading
import pandas as pd
import time
import numpy as np
import yt_dlp
from yt_dlp import DownloadError, SameFileError

processed = 0
gold = 0
silver = 0
start = time.time()
lock = threading.Lock()

_real_stdout = sys.stdout
_real_stderr = sys.stderr

def _print(*args, **kwargs):
    print(*args, **kwargs, file=_real_stdout)

sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    results = pd.read_csv('status.csv', index_col=False)
    _print('Continuing...')
except:
    results = pd.DataFrame(columns=['videoID', 'status', 'details'])

processed_last_session = len(results['videoID'])

def download_en_captions(vid_id):
    url = f'https://www.youtube.com/watch?v={vid_id}'

    try:
        yt_dlp._real_main([ "-4", "-o", "sponsor/%(id)s.%(ext)s","--skip-download","--write-sub",
                "--sub-lang", "en,en-AU,en-BZ,en-CA,en-IE,en-JM,en-NZ,en-ZA,en-TT,en-GB,en-US", url ])
    except Exception:
        pass
    except SystemExit:
        pass

    files = glob(f'sponsor/{vid_id}.*')
    if len(files) > 0:
        for file in files[1:]:
            os.remove(file)
        return 'gold', ''
    else:
        error = None
        try:
            yt_dlp._real_main([ "-4", "-o", "sponsor/%(id)s.%(ext)s","--skip-download","--write-auto-sub",
                "--sub-lang", "en", url])
        except DownloadError:
            error = f'Download error!'
        except SameFileError as e:
            error = str(e)
        except SystemExit:
            pass

        files = glob(f'sponsor/{vid_id}.*')
        if len(files) > 0:
            return 'silver', ''
        else:
            return 'none', error

def process_video(vid_id):
    global gold
    global silver
    global processed

    try:
        status, details = download_en_captions(vid_id)
        if status == 'gold':
            gold += 1
        if status == 'silver':
            silver += 1
        with lock:
            results.loc[len(results.index)] = [vid_id, status, details]
    except Exception as e:
        _print(e)
        return
    processed += 1

sponsor_times = pd.read_csv('sponsorVideoIDs_1646386033207.csv', sep=',', index_col=False)
video_ids = sponsor_times['videoID'].unique()
np.random.shuffle(video_ids)
video_ids = set(video_ids)

_print(f'Found {len(video_ids)} videos')

if processed_last_session > 0:
    _print(f'{processed_last_session} videos already processed.');
    video_ids = video_ids.difference(set(results['videoID']))

_print(f'{len(video_ids)} videos remaining')

p = multiprocessing.dummy.Pool(processes=32)

start = time.time()
def _print_progress():
    while True:
        if processed == 0:
            _print('.', end='')
            time.sleep(1)
            continue
        elapsed = time.time() - start
        _print(f"""{processed} videos processed at a rate of {(elapsed / processed):.2f}s per video
            {(gold / processed * 100):.2f}% of videos have GOLD captions, {(silver / processed * 100):.2f}% of videos have SILVER captions
            {((processed + processed_last_session) / len(video_ids) * 100):.2f}% done""")
        with lock:
            results.to_csv('status.csv', index=False)
        time.sleep(5)

timer = threading.Thread(target=_print_progress)
timer.start()

p.map(process_video, video_ids)

p.close()
p.join()
