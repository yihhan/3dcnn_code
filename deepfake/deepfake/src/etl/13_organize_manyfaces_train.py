import pandas as pd
import numpy as np


with open('../../data/dfdc/manyfaces_files_list.txt') as f:
    manyfaces = [line.strip() for line in f.readlines()]


df = pd.read_csv('../../data/dfdc/train_with_splits.csv')

faces_df = pd.DataFrame({
        'imgfile': [_.replace('../../data/dfdc/', '') for _ in manyfaces]
    })

faces_df['folder'] = [_.split('/')[1] for _ in faces_df['imgfile']]
faces_df['filename'] = [_.split('/')[2] + '.mp4' for _ in faces_df['imgfile']]

train_df = df[df['split'] == 'train']
valid_df = df[df['split'] == 'valid']

train_df = train_df.merge(faces_df, on=['folder', 'filename'])
valid_df = valid_df.merge(faces_df, on=['folder', 'filename'])
# Only 2 videos in validation set had no faces ...
# So we'll just overestimate our performance by ignoring them

df = pd.concat([train_df, valid_df])
df = df[(df['original'].isin(df['filename'])) | (df['label'] == 0)]

df.to_csv('../../data/dfdc/train_manyfaces_with_splits.csv', index=False)