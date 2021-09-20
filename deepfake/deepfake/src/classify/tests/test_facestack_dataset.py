import factory.builder as builder

import pandas as pd
import numpy as np
import yaml

from run import get_stacked_labels 


with open('configs/experiments/experiment010.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


cfg['dataset']['name'] = 'FaceSeqDataset'

df = pd.read_csv('../../data/dfdc/train_manyfaces_with_splits.csv')
train_df = df[df['split'] == 'train']

train_images, train_labels, train_df = get_stacked_labels(train_df, cfg)

train_loader = builder.build_dataloader(cfg, data_info={'imgfiles': train_images, 'labels': train_labels}, train_images=train_images, train_df=train_df, mode='train')

loader = iter(train_loader)
data = next(loader)

