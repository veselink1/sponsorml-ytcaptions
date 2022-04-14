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

# File with column 'videoID'
INPUT_FILENAME = 'sponsorVideoIDs_1646386033207.csv'
# File to use to save job state
STATUS_FILENAME = 'status.csv'
# Directory to store downloaded captions in
OUTPUT_PATH = 'sponsor'
# Number of concurrent yt-dlp scripts to run
NUM_THREADS = 32

# Runtime stats to display progress
processed = 0
gold = 0
silver = 0
start = time.time()
# Synchronisation for the variables above and the STATUS_FILENAME
lock = threading.Lock()

# Save the output streams -- we redirect sys.stdout/stderr to /dev/null
# to hide the output from yt_dlp
_real_stdout = sys.stdout
_real_stderr = sys.stderr

# Convenience wrapper for print which prints to the real stdout handle
def _print(*args, **kwargs):
    print(*args, **kwargs, file=_real_stdout)

# Pipe to /dev/null
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    # Try to restore the state from last time
    results = pd.read_csv(STATUS_FILENAME, index_col=False)
    _print('Continuing...')
except:
    results = pd.DataFrame(columns=['videoID', 'status', 'details'])

processed_last_session = len(results['videoID'])

def download_en_captions(vid_id, output_path):
    """
    Downloads the English language captions for the video.

    Returns the tuple `(status: 'gold'|'silver'|'none', message: str)`
    """
    url = f'https://www.youtube.com/watch?v={vid_id}'

    try:
        # Call into yt_dlp
        yt_dlp._real_main([
            # Use IPv4
            "-4",
            # Output filename template
            "-o", output_path + "/%(id)s.%(ext)s",
             # Do not download the video itself
            "--skip-download",
            # Download all available English language captions
            "--write-sub",
            "--sub-lang", "en,en-AU,en-BZ,en-CA,en-IE,en-JM,en-NZ,en-ZA,en-TT,en-GB,en-US",
            url
        ])
    except Exception:
        # We detect failure by checking for the file itself.
        pass
    except SystemExit:
        # When the download fails, yt_dlp calls exit(code).
        pass

    # Check for the output files (if any).
    files = glob(f'{output_path}/{vid_id}.*')
    if len(files) > 0:
        # Keep only the first result (in alphabetical order ~ *.en.*)
        for file in files[1:]:
            os.remove(file)
        return 'gold', ''
    else:
        # Nothing was downloaded, we still need to check for
        # auto-generated captions.
        error = None
        try:
            # Use the same options as above
            yt_dlp._real_main([ "-4", "-o", output_path + "/%(id)s.%(ext)s","--skip-download","--write-auto-sub",
                "--sub-lang", "en", url])
        except DownloadError:
            error = f'Download error!'
        except SameFileError as e:
            error = str(e)
        except SystemExit:
            pass

        files = glob(f'{output_path}/{vid_id}.*')
        if len(files) > 0:
            return 'silver', ''
        else:
            return 'none', error

def process_video(vid_id, output_path):
    """
    Downloads the captions for the video to the output
    path save the job state to disk.
    """
    global gold
    global silver
    global processed

    try:
        status, details = download_en_captions(vid_id, output_path)
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

sponsor_times = pd.read_csv(INPUT_FILENAME, sep=',', index_col=False)
video_ids = sponsor_times['videoID'].unique()
np.random.shuffle(video_ids)
video_ids = set(video_ids)

_print(f'Found {len(video_ids)} videos')

if processed_last_session > 0:
    _print(f'{processed_last_session} videos already processed.');
    video_ids = video_ids.difference(set(results['videoID']))

_print(f'{len(video_ids) - processed_last_session} videos remaining')

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
            results.to_csv(STATUS_FILENAME, index=False)
        time.sleep(5)

timer = threading.Thread(target=_print_progress)
timer.start()

p.map(lambda vid_id: process_video(vid_id, OUTPUT_PATH), video_ids)

p.close()
p.join()
