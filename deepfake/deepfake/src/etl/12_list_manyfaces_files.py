import glob
import os, os.path as osp

from tqdm import tqdm


MANYFACES = '../../data/dfdc/manyfaces/'

parts = os.listdir(MANYFACES)

videos = []
for p in tqdm(parts, total=len(parts)):
    videos_in_part = os.listdir(osp.join(MANYFACES, p))
    videos.extend([osp.join(MANYFACES, p, v) for v in videos_in_part])


faces = []
for v in tqdm(videos, total=len(videos)):
    faces_in_video = os.listdir(v)
    #faces.extend([osp.join(v, f) for f in faces_in_video])
    faces.extend(glob.glob(osp.join(v, '*.png')))


with open('../../data/dfdc/manyfaces_files_list.txt', 'w') as f:
    for _ in faces:
        status = f.write('{}\n'.format(_))