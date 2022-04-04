import pandas as pd

# Filters the csv so that only the videoID column is preserved.
sponsor_times = pd.read_csv('sponsorTimes_1646386033207.csv', sep=',')
video_ids = sponsor_times[['videoID']]
video_ids = video_ids[sponsor_times['category'] == 'sponsor']
video_ids.to_csv('sponsorVideoIDs_1646386033207.csv')
