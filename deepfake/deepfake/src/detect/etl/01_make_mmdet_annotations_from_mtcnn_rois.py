import pandas as pd
import numpy as np
import pickle
import cv2
import os.path as osp
import imagesize

from tqdm import tqdm


DATADIR = '.'
FRAMEDIR = 'frame-mtcnn/'
df = pd.read_csv(osp.join(DATADIR, 'cleaned_mtcnn_rois.csv'))

annotations = []
frame_shape_dict = {
    fp : imagesize.get(osp.join(FRAMEDIR, fp)) for fp in tqdm(np.unique(df['imgfile']), total=len(np.unique(df['imgfile'])))
}
dfgroups = list(df.groupby('imgfile'))
for _fp, _df in tqdm(dfgroups, total=len(dfgroups)):
    width, height = frame_shape_dict(fp)
    boxes = np.asarray(_df[['x1','y1','x2','y2']])
    tmp_dict = {
        'filename': _fp,
        'height': height,
        'width':  width,
        'ann': {
            'bboxes': boxes,
            'labels': np.asarray([1] * len(_df))
        },
        'img_class': 1,
        'split': _df['split'].iloc[0]
    }
    assert len(tmp_dict['ann']['bboxes']) == len(tmp_dict['ann']['labels'])
    annotations.append(tmp_dict)

with open(osp.join(DATADIR, 'train_from_mtcnn_bbox_annotations_with_splits.pkl'), 'wb') as f:
    pickle.dump(annotations, f)