import numpy as np
import pickle
import cv2
import os, os.path as osp
import gc

from tqdm import tqdm
from ensemble_boxes import GeneralEnsemble, getCoords
from dataset import FramesDataset
from helper import scale_bbox, enlarge_box, convert_coords, intify, get_frames
from torch.utils.data import DataLoader




# Load face predictions
with open('../../data/dfdc/face_predictions_round2/predictions.pkl', 'rb') as f:
    face_preds = pickle.load(f)

THRESHOLD = 0.55
VIDEODIR = '../../data/dfdc/videos/'
FACESDIR = '../../data/dfdc/manyfaces/'

boxes = face_preds['y_pred']
names = face_preds['names']

# Sort by name
sorted_indices = np.argsort(names)
boxes = np.asarray(boxes)[sorted_indices]
boxes = [b[0] for b in boxes]
names = np.asarray(names)[sorted_indices]

face_preds = []
# Refactor face predictions so that each element represents a video
# We know there are 3 frames per video
for ind in range(0, len(names), 3):
    tmp_dict = {
        'vidname': names[ind].split('-')[0]+'.mp4',
        'bboxes': {
            0 : boxes[ind],
            1 : boxes[ind+1],
            2 : boxes[ind+2]
        }
    }
    face_preds.append(tmp_dict)

# Use if this script gets interrupted/throws an error
# continue_from = ...
# face_preds = face_preds[continue_from:]

dset = FramesDataset(videos=face_preds, video_dir=VIDEODIR)
# loader = DataLoader(dset, batch_size=1, num_workers=8, shuffle=False)
# loader = iter(loader)

for index in tqdm(range(26830, len(face_preds)), total=len(face_preds)):
    vid = face_preds[index]
    savedir = osp.join(FACESDIR, vid['vidname'].replace('.mp4', ''))
    if not osp.exists(savedir): os.makedirs(savedir)
    #
    tmp_bboxes = {
        i : vid['bboxes'][i][vid['bboxes'][i][:,-1] >= THRESHOLD]
        for i in [0,1,2]
    }
    tmp_bboxes = [v[:,:-1] for k,v in tmp_bboxes.items() if v.shape[0] > 0]
    if len(tmp_bboxes) == 0: 
        print ('No faces detected for {} !'.format(vid['vidname']))
        continue
    #
    # Scale up
    #tmp_bboxes = [scale_bbox(b, scale_factor) for b in tmp_bboxes]
    # Enlarge
    #tmp_bboxes = [enlarge_box(b, frames[0].shape, scale=1.1) for b in tmp_bboxes]
    # Convert to [xc, yc, w, h] format for ensemble box function
    tmp_bboxes = [convert_coords(b) for b in tmp_bboxes]
    # If there is more than 1 box detected in a frame, merge boxes with IoU >= 0.5 in a frame
    tmp_bboxes = [[list(b)] for b in tmp_bboxes]
    tmp_bboxes = [GeneralEnsemble(box_list, iou_thresh=0.5) for box_list in tmp_bboxes]
    # Then, merge across frames
    merged_bbox = GeneralEnsemble(tmp_bboxes)
    # Convert back to [x1, y1, x2, y2]
    # Only if at least 2/3 frames have the box
    merged_bbox = [getCoords(box[:-2]) for box in merged_bbox if box[-1] >= 0.66]
    # Now, what if the person moves throughout the frame? 
    # Then, boxes across frames might not overlap
    # So combine them ...
    if len(merged_bbox) == 0:
        frames = np.asarray(dset[index])
        #frames = np.asarray([_.squeeze(0).numpy() for _ in frames])
        if frames.shape == (1,):
            continue
        #
        scale_factor = np.max(frames[0].shape) / 640.
        tmp_bboxes = {
            i : vid['bboxes'][i][vid['bboxes'][i][:,-1] >= THRESHOLD]
            for i in [0,1,2]
        }
        tmp_bboxes = [v[:,:-1] for k,v in tmp_bboxes.items() if v.shape[0] > 0]
        if len(tmp_bboxes) == 0: 
            print ('No faces detected for {} !'.format(vid['vidname']))
            continue
        #
        # Scale up
        tmp_bboxes = [scale_bbox(b, scale_factor) for b in tmp_bboxes]
        # Enlarge
        tmp_bboxes = [enlarge_box(b, frames[0].shape, scale=1.1) for b in tmp_bboxes]
        # Convert to [xc, yc, w, h] format for ensemble box function
        tmp_bboxes = [convert_coords(b) for b in tmp_bboxes]
        # If there is more than 1 box detected in a frame, merge boxes with IoU >= 0.5 in a frame
        tmp_bboxes = [[list(b)] for b in tmp_bboxes]
        tmp_bboxes = [GeneralEnsemble(box_list, iou_thresh=0.5) for box_list in tmp_bboxes]
        # Then, merge across frames
        merged_bbox = GeneralEnsemble(tmp_bboxes)
        # Convert back to [x1, y1, x2, y2]
        # Only if at least 2/3 frames have the box
        merged_bbox = [getCoords(box[:-2]) for box in merged_bbox if box[-1] >= 0.66]
        #
        new_tmp_bboxes = []
        # Deal with uneven # of boxes per frame
        min_len = np.min([len(b) for b in tmp_bboxes])
        tmp_bboxes = [b[:min_len] for b in tmp_bboxes]
        for box in tmp_bboxes:
            new_tmp_bboxes.append([getCoords(b[:-2]) for b in box])
        tmp_bboxes = np.asarray(new_tmp_bboxes)
        merged_bbox = [[np.min(tmp_bboxes[...,0]), np.max(tmp_bboxes[...,1]), np.min(tmp_bboxes[...,2]), np.max(tmp_bboxes[...,3])]]
    else:
        continue
    # getCoords returns [x1, x2, y1, x2]
    merged_bbox = [(box[0], box[2], box[1], box[3]) for box in merged_bbox]
    merged_bbox = [intify(box) for box in merged_bbox]
    assert len(merged_bbox) > 0
    frames = np.asarray(frames)
    faces = [frames[:,y1:y2,x1:x2,:] for x1,y1,x2,y2 in merged_bbox]
    for face_ind, face in enumerate(faces):
        for fr_ind, fr in enumerate(face):
            status = cv2.imwrite(osp.join(savedir, 'FRAME{:02d}-FACE{}.png'.format(fr_ind, face_ind)), fr)



# for vid in tqdm(face_preds, total=len(face_preds)):
#     vidfile = osp.join(VIDEODIR, vid['vidname'])
#     savedir = osp.join(FACESDIR, vid['vidname'].replace('.mp4', ''))
#     if not osp.exists(savedir): os.makedirs(savedir)

#     frames, _ = get_frames(vidfile, num_frames=21, resize=None)
#     scale_factor = np.max(frames[0].shape) / 640.
#     tmp_bboxes = {
#         i : vid['bboxes'][i][vid['bboxes'][i][:,-1] >= THRESHOLD]
#         for i in [0,1,2]
#     }
#     tmp_bboxes = [v[:,:-1] for k,v in tmp_bboxes.items() if v.shape[0] > 0]
#     if len(tmp_bboxes) == 0: 
#         print ('No faces detected for {} !'.format(vidfile))
#         continue

#     # Scale up
#     tmp_bboxes = [scale_bbox(b, scale_factor) for b in tmp_bboxes]
#     # Enlarge
#     tmp_bboxes = [enlarge_box(b, frames[0].shape, scale=1.1) for b in tmp_bboxes]
#     # Convert to [xc, yc, w, h] format for ensemble box function
#     tmp_bboxes = [convert_coords(b) for b in tmp_bboxes]
#     # If there is more than 1 box detected in a frame, merge boxes with IoU >= 0.5 in a frame
#     tmp_bboxes = [[list(b)] for b in tmp_bboxes]
#     tmp_bboxes = [GeneralEnsemble(box_list, iou_thresh=0.5) for box_list in tmp_bboxes]
#     # Then, merge across frames
#     merged_bbox = GeneralEnsemble(tmp_bboxes)
#     # Convert back to [x1, y1, x2, y2]
#     # Only if at least 2/3 frames have the box
#     merged_bbox = [getCoords(box[:-2]) for box in merged_bbox if box[-1] >= 0.66]
#     # getCoords returns [x1, x2, y1, x2]
#     merged_bbox = [(box[0], box[2], box[1], box[3]) for box in merged_bbox]
#     merged_bbox = [intify(box) for box in merged_bbox]
#     frames = np.asarray(frames)
#     faces = [frames[:,y1:y2,x1:x2,:] for x1,y1,x2,y2 in merged_bbox]
#     for face_ind, face in enumerate(faces):
#         for fr_ind, fr in enumerate(face):
#             status = cv2.imwrite(osp.join(savedir, 'FRAME{:02d}-FACE{}.png'.format(fr_ind, face_ind)), fr)





