import matplotlib
import pandas as pd

matplotlib.use('agg')
from os.path import join
from collections import Counter

metadata1 = pd.read_csv(join('..', 'data', 'subject_data.csv'))
metadata1['Subnum'] = metadata1['partNum']
metadata1 = metadata1[['Subnum', 'Age', 'Sex']].drop_duplicates()

metadata2 = pd.read_csv(join('..', 'data', 'metadata.csv'))
metadata2['Subnum'] = metadata2['Subnum'].apply(lambda s: int(s[2:]))

raw_data = pd.read_csv(join('..', 'data', 'merged_data_all.csv'))
participants = raw_data[['participant']].drop_duplicates().sort_values(['participant'])

metadata = pd.concat([metadata1, metadata2]) \
    .drop_duplicates() \
    .sort_values(['Subnum'])  # type: pd.DataFrame
c = Counter(metadata['Subnum'])

print("mismatched duplicates:")
for k, v in c.items():
    if v > 1:
        print(k, v)

merged = metadata.merge(participants, left_on="Subnum", right_on="participant")[
    metadata.columns.values].drop_duplicates(['Subnum'])  # type: pd.DataFrame
merged['IsFemale'] = merged['Sex'].apply(lambda s: int(s == 'F'))

print(merged.describe())
