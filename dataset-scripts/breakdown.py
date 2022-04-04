import os
import threading
import pandas as pd
import numpy as np
import time

def get_all_ccs(videos):
    base_url = 'https://www.youtube.com/watch?v='
    lang="en"
    for vid in vids:
        url = base_url + vid
        cmd = ["youtube-dl","--skip-download","--write-sub",
               "--sub-lang",lang,url]
        os.system(" ".join(cmd))

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def download_eng_captions(vid_id):
    # print(f'Processing {vid_id}')
    url = f'https://www.youtube.com/watch?v={vid_id}'
    cmd = ["youtube-dl", "-o", "captions/%(id)s.%(ext)s","--skip-download","--write-sub",
            "--sub-lang", "en", url]
    os.system(" ".join(cmd))


sponsor_times = pd.read_csv('sponsorTimes_1646386033207.csv', sep=',')

number_of_entries = len(sponsor_times['category'])

for category in set(sponsor_times['category']):
    count = np.count_nonzero(sponsor_times['category'] == category)
    print(f'{category}: {count}/{number_of_entries} ({count / number_of_entries * 100})')
