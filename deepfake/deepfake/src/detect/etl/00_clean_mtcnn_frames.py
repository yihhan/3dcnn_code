import pandas as pd
import glob

df = pd.read_csv('mtcnn_frame_face_rois.csv')

with open('frames_to_use.txt') as f:
    frames = [_.strip() for _ in f.readlines()]


frames_df = pd.DataFrame({'framefile': frames})
frames_df['num_face'] = [int(_.split('_')[-1].replace('.png', '')) for _ in frames_df['framefile']]
frames_df['vidname'] = [_.split('_')[0] for _ in frames_df['framefile']]
df['vidname'] = [_.split('/')[-1] for _ in df['filename']]

df = df.merge(frames_df, on=['vidname', 'num_face'])

frame_images = glob.glob('frame-mtcnn/*/*.png')
frame_images = ['/'.join(_.split('/')[1:]) for _ in frame_images]
frame_image_df = pd.DataFrame({'imgfile': frame_images})
df['imgfile'] = [_.replace('mp4','png') for _ in df['filename']]

df = df.merge(frame_image_df, on='imgfile')

df.to_csv('cleaned_mtcnn_rois.csv', index=False)