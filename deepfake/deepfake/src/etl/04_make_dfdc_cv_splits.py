import pandas as pd

df = pd.read_csv('../../data/dfdc/train.csv')

# Let's do single train/valid split
# First and last 4 chunks as valid

df['split'] = 'train'
df['part'] = [int(_.split('_')[-1]) for _ in df['folder']]

df.loc[df['part'].isin([0, 46,47,48,49]), 'split']  = 'valid'

# Subsample valid to 1:1 
train_df = df[df['split'] == 'train']
valid_df = df[df['split'] == 'valid']
df_list = []
for p in valid_df['part'].unique():
    part_real_valid_df = valid_df[(valid_df['part'] == p) & (valid_df['label'] == 'REAL')]
    part_fake_valid_df = valid_df[(valid_df['part'] == p) & (valid_df['label'] == 'FAKE')]
    part_fake_valid_df = part_fake_valid_df.sample(n=len(part_real_valid_df), replace=False, random_state=0)
    df_list.extend([part_real_valid_df, part_fake_valid_df])

df = pd.concat([train_df] + df_list)
df = df.sample(frac=1).reset_index(drop=True)

df['label'] = (df['label'] == 'FAKE').astype('float32')
df.to_csv('../../data/dfdc/train_with_splits.csv', index=False)
