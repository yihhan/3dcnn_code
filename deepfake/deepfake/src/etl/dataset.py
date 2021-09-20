import cv2
import numpy as np
import gc
import os, os.path as osp

from torch.utils.data import Dataset


class FramesDataset(Dataset):

    def __init__(self, 
                 videos,
                 video_dir):

        self.videos = videos
        self.video_dir = video_dir

    def __len__(self):
        return len(self.videos)

    @staticmethod
    def get_frames(vidfile, num_frames=-1, resize=None):
        cap = cv2.VideoCapture(vidfile)
        ret, frame = cap.read()
        # Longest side = resize
        if resize:
            if np.argmax(frame.shape) == 0:
                resize_hw = (resize, int(frame.shape[1] * resize / frame.shape[0]))
            elif np.argmax(frame.shape) == 1:
                resize_hw = (int(frame.shape[0] * resize / frame.shape[1]), resize)
            frame = cv2.resize(frame, resize_hw[::-1])
        frames = [frame]
        while ret and len(frames) < num_frames:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, resize_hw[::-1]) if resize else frame
                frames.append(frame)
        cap.release()
        gc.collect()
        return frames, list(range(num_frames))

    def __getitem__(self, i):
        vidfile = self.videos[i]['vidname']
        try:
            frames, _ = self.get_frames(osp.join(self.video_dir, vidfile), num_frames=21)
            if frames[0] is None:
                return 0
            else:
                return frames
        except:
            return 0
