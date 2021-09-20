import pickle
import glob
import cv2
import os, os.path as osp

from tqdm import tqdm


DATADIR = '../../data/dfdc/frames/'
PICKLEPATH = '../../data/dfdc/frames_annotations_for_widerface_retinanet.pkl'

all_frames = glob.glob(osp.join(DATADIR, 'dfdc_train_part*', '*.png'))

if osp.exists(PICKLEPATH):
    with open(PICKLEPATH, 'rb') as f:
        annotations = pickle.load(f)
        alreadydone = [_['filename'] for _ in annotations]
else:
    annotations = []
    alreadydone = []


for frame in tqdm(all_frames, total=len(all_frames)):
    fp = '/'.join(frame.split('/')[-2:])
    if fp in alreadydone:
        continue
    img = cv2.imread(frame)
    tmp_dict = {
        'filename': fp,
        'height': img.shape[0],
        'width':  img.shape[1]
    }
    annotations.append(tmp_dict)

with open(PICKLEPATH, 'wb') as f:
    pickle.dump(annotations, f)