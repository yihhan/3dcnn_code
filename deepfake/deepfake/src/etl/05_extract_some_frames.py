import cv2
import pandas as pd
import numpy as np
import torch
import os, os.path as osp
import gc

from tqdm import tqdm


def get_frames(vidfile, num_frames=-1, resize=None):
    cap = cv2.VideoCapture(vidfile)
    ret, frame = cap.read()
    # Longest side = resize
    if np.argmax(frame.shape) == 0:
        resize_hw = (resize, int(frame.shape[1] * resize / frame.shape[0]))
    elif np.argmax(frame.shape) == 1:
        resize_hw = (int(frame.shape[0] * resize / frame.shape[1]), resize)
    frame = cv2.resize(frame, resize_hw[::-1]) if resize else frame
    frames = [frame]
    while ret and len(frames) < num_frames:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, resize_hw[::-1]) if resize else frame
            frames.append(frame)
    cap.release()
    gc.collect()
    return frames, list(range(num_frames))


# Load in the first M frames
# Save every Nth frame
# Thus each video will produce M/N frames, which we will save as images
# Extracting too many frames would overwhelm storage

VIDEODIR = '../../data/dfdc/videos/'
FRAMEDIR = '../../data/dfdc/frames/'

if not osp.exists(FRAMEDIR):
    os.makedirs(FRAMEDIR)


df = pd.read_csv('../../data/dfdc/train.csv')

NUMFRAMES = 25
STAGGER = 5

for rownum in tqdm(range(len(df)), total=len(df)):
    frames, inds = get_frames(osp.join(VIDEODIR, df['filepath'].iloc[rownum]),
                                num_frames=NUMFRAMES,
                                resize=640)
    frames_sample, inds_sample = frames[::STAGGER], inds[::STAGGER]
    assert len(frames_sample) == len(inds_sample) == NUMFRAMES // STAGGER
    PARTDIR = df['folder'].iloc[rownum]
    SAVEDIR = osp.join(FRAMEDIR, PARTDIR)
    if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)
    video_name = df['filename'].iloc[rownum].replace('.mp4', '')
    for idx, frame in enumerate(frames_sample):
        status = cv2.imwrite(osp.join(SAVEDIR, '{}-{:02d}.png'.format(video_name, inds_sample[idx])), frame)


