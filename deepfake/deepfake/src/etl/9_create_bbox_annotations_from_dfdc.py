import pandas as pd
import numpy as np
import pickle
import re
import os.path as osp

from tqdm import tqdm

# Load in train/valid split for DFDC
split_df = pd.read_csv('../../data/dfdc/train_with_splits.csv')
# Load in bounding box data from faces_to_examine
faces_df = pd.read_csv('../../data/dfdc/faces_to_examine.csv')
# Load in bounding box data from examined faces
# Only when >1 face was detected was the face examined
examined_df = pd.read_csv('../../data/dfdc/faces_to_examine/examined_faces.csv')
# Load in annotations when predicting on DFDC data
# This will save time, since we won't have to load the images again
with open('../../data/dfdc/frames_annotations_for_widerface_retinanet.pkl', 'rb') as f:
    frames_ann = pickle.load(f)

# Turn it into a dict
frames_ann_dict = {_['filename'] : (_['height'], _['width']) for _ in frames_ann}

# Load in DFDC predictions to get number of faces detected
# per frame
with open('../../data/dfdc/face_predictions/predictions.pkl', 'rb') as f:
    face_preds = pickle.load(f)


boxes = face_preds['y_pred']
names = face_preds['names']

THRESHOLD = 0.6

num_faces = []
for b in tqdm(boxes, total=len(boxes)):
    b = b[0]
    num_faces.append(b[b[:,-1] >= THRESHOLD].shape[0])

# Make num_face_df to merge with faces_df
num_face_df = pd.DataFrame({
        'num_faces': num_faces,
        '_imgfile': names
    })
faces_df['imgfile'] = [_.replace('../../data/dfdc/faces_to_examine/', '') for _ in faces_df['imgfile']]
faces_df['_imgfile'] = ['-'.join(_.split('-')[:-1])+'.png' for _ in faces_df['imgfile']]
faces_df = faces_df.merge(num_face_df, on='_imgfile')

one_face_df = faces_df[faces_df['num_faces'] == 1]
two_face_df = faces_df[faces_df['num_faces'] >  1]
# Use examined_df to update two_face_df
examined_df['imgfile'] = [re.sub(r'facex[0-9]/', '', _) for _ in examined_df['imgfile']] 
two_face_df = two_face_df.merge(examined_df, on='imgfile')
faces_df = pd.concat([one_face_df, two_face_df])
faces_df['filename'] = [_.split('/')[-1].split('-')[0] + '.mp4' for _ in faces_df['_imgfile']]
faces_df = faces_df.merge(split_df, on='filename', how='left')
# Some videos are missing because I subsample validation to be 50/50
# Assign those to be valid
faces_df.loc[faces_df['split'].isna(), 'split'] = 'valid'
faces_df['framefile'] = ['-'.join(_.split('-')[:-1])+'.png' for _ in faces_df['imgfile']]

# Now, make the annotations
def enlarge_box(box, img_shape, scale=1.1):
    # box = (x1, y1, x2, y2)
    # w = max width
    # h = max height
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    w *= scale
    h *= scale
    xc = (x2 + x1) / 2
    yc = (y2 + y1) / 2
    x1 = np.max((0, xc - w / 2))
    y1 = np.max((0, yc - h / 2))
    x2 = np.min((img_shape[1], xc + w / 2))
    y2 = np.min((img_shape[0], yc + h / 2))    
    return int(x1), int(y1), int(x2), int(y2)


annotations = []
for _fp, _df in tqdm(faces_df.groupby('framefile'), total=len(faces_df['framefile'].unique())):
    height, width = frames_ann_dict[_fp]
    boxes = np.asarray(_df[['x1','y1','x2','y2']])
    boxes = np.asarray([enlarge_box(_, (height,width), 1.5) for _ in boxes])
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

with open('../../data/dfdc/train_bbox_annotations_with_splits.pkl', 'wb') as f:
    pickle.dump(annotations, f)

abbrev_frames_ann = [_ for _ in frames_ann if _['filename'].split('-')[-1].replace('.png', '') in ('00','10','20')]

with open('../../data/dfdc/abbrev_frames_annotations_for_widerface_retinanet.pkl', 'wb') as f:
    pickle.dump(abbrev_frames_ann, f)

