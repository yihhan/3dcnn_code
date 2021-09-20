import pandas as pd
import numpy as np
import pickle
import cv2
import glob
import os, os.path as osp

from tqdm import tqdm


PICKLEPATH = '../../data/dfdc/face_predictions/predictions.pkl'

with open(PICKLEPATH, 'rb') as f:
    face_preds = pickle.load(f)

boxes = face_preds['y_pred']
names = face_preds['names']

THRESHOLD = 0.6

EXAMINEDIR = '../../data/dfdc/faces_to_examine/'

num_faces = []
for b in tqdm(boxes, total=len(boxes)):
    b = b[0]
    num_faces.append(b[b[:,-1] >= THRESHOLD].shape[0])

values, counts = np.unique(num_faces, return_counts=True)

# Check to see highest score in frames where no face meets the threshold
high_scores = []
for num, box in zip(num_faces, boxes):
    if num == 0: 
        try:
            high_scores.append(box[0][0,-1])
        except IndexError:
            high_scores.append(0)

# Organize faces into folders within parts based on number of faces
# in that frame
num_face_dict = {}
for nf, name in zip(num_faces, names):
    num_face_dict[name.replace('.png', '')] = nf

# Make directories in each dfdc_part for number of faces
dfdc_parts = glob.glob(osp.join(EXAMINEDIR, '*'))
for part in dfdc_parts:
    for v in values:
        num_face_directory = osp.join(part, 'facex{}'.format(v))
        if not osp.exists(num_face_directory):
            print('Creating {} ...'.format(num_face_directory))
            os.makedirs(num_face_directory)
        else:
            print('{} already exists !'.format(num_face_directory))


# Handle frames without detected faces by copying frame image
# to facex0
for num, name in zip(num_faces, names):
    if num == 0:
        copyfile = osp.join('../../data/dfdc/frames', name)
        destination = osp.join(EXAMINEDIR, name.split('/')[0], 'facex0')
        status = os.system('cp {} {}'.format(copyfile, destination))


all_face_imgs = np.sort(glob.glob(osp.join(EXAMINEDIR, '*/*.png')))

for each_face in tqdm(all_face_imgs, total=len(all_face_imgs)):
    prefix = '-'.join('/'.join(each_face.split('/')[-2:]).split('-')[:2])
    destination = osp.join(EXAMINEDIR, prefix.split('/')[0], 'facex{}'.format(num_face_dict[prefix]))
    status = os.system('mv {} {}'.format(each_face, destination))