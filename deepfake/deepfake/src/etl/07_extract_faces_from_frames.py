import pandas as pd
import numpy as np
import pickle
import cv2
import os, os.path as osp

from tqdm import tqdm


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


PICKLEPATH = '../../data/dfdc/face_predictions/predictions.pkl'

with open(PICKLEPATH, 'rb') as f:
    face_preds = pickle.load(f)


boxes = face_preds['y_pred']
names = face_preds['names']


Tabulate number of faces per image based on threshold
thresholds = np.arange(0.40, 1.00, 0.05)
num_faces = {t : [] for t in thresholds}

for b in tqdm(boxes, total=len(boxes)):
    b = b[0]
    for t in thresholds:
        num_faces[t].append(b[b[:,-1] >= t].shape[0])

for k,v in num_faces.items():
    values, counts = np.unique(v, return_counts=True)
    print('Threshold {:.2f}'.format(k))
    print('--------------')
    for val,count in zip(values, counts):
        print('{} : n={}'.format(val,count))
    print('\n')

# Let's use threshold 0.6 for now
face_dict = {
    'x1': [],
    'y1': [],
    'x2': [],
    'y2': [],
    'imgfile': []
}
DATADIR = '../../data/dfdc/frames/'
SAVEDIR = '../../data/dfdc/faces_to_examine/'
THRESHOLD = 0.6
for box, name in tqdm(zip(boxes, names), total=len(boxes)):
    suffix = name.split('/')[-1].split('-')[-1].replace('.png', '')
    if suffix not in ['00', '20']:
        continue
    folder = osp.join(SAVEDIR, name.split('/')[0])
    if not osp.exists(folder): os.makedirs(folder)
    frame = cv2.imread(osp.join(DATADIR, name))
    box = box[0]
    box = box[box[:,-1] >= THRESHOLD]
    for i,b in enumerate(box):
        x1, y1, x2, y2, p = b
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_dict['x1'].append(x1)
        face_dict['y1'].append(y1)
        face_dict['x2'].append(x2)
        face_dict['y2'].append(y2)
        x1, y1, x2, y2 = enlarge_box([x1,y1,x2,y2], frame.shape, scale=3)
        face = frame[y1:y2, x1:x2]
        savefile = osp.join(folder, '{}-{:02d}.png'.format(name.split('/')[-1].replace('.png', ''), i))
        face_dict['imgfile'].append(savefile)
        status = cv2.imwrite(savefile, face)


face_df = pd.DataFrame(face_dict)
face_df.to_csv('../../data/dfdc/faces_to_examine.csv', index=False)
