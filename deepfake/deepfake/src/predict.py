import argparse
import torch
import pandas as pd
import numpy as np
import yaml
import glob
import cv2
import os, os.path as osp

from tqdm import tqdm
from torch.utils.data import DataLoader
from mmcv.parallel import collate 

from etl.dataset import FramesDataset
from etl.helper import *
from etl.ensemble_boxes import GeneralEnsemble, getCoords
from detect.factory.reproducibility import set_reproducibility
import detect.factory.builder as Dbuilder
import detect.factory.evaluate as Dfactory_evaluate
from classify.factory.models import *
import classify.factory.builder as Cbuilder
import classify.factory.evaluate as Cfactory_evaluate

def predict(model, data):
    if type(data['img']) != list and type(data['img_meta']) != list:
        data['img'] = [data['img']]
        data['img_meta'] = [data['img_meta']]
    data_gpu = collate([data], samples_per_gpu=1)
    with torch.no_grad():
        output = model(**data_gpu, return_loss=False, rescale=True)
    return output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=-1)
    return parser.parse_args()

args = parse_args()

with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

if args.num_workers > 0:
    cfg['num_workers'] = args.num_workers

# We will set all the seeds we can, in vain ...
set_reproducibility(cfg['seed'])
# Set GPU
torch.cuda.set_device(args.gpu)

with open(cfg['detect']) as f:
    Dcfg = yaml.load(f, Loader=yaml.FullLoader)

with open(cfg['classify']) as f:
    Ccfg = yaml.load(f, Loader=yaml.FullLoader)

VIDEODIR = cfg['video_dir']
videos = glob.glob(osp.join(VIDEODIR, '*.mp4'))

# Loader for grabbing frames from videos
dset = FramesDataset(videos=[{'vidname': _} for _ in videos], 
                     video_dir='')
loader = DataLoader(dset, 
                    batch_size=1, 
                    num_workers=cfg['num_workers'], 
                    shuffle=False)
loader = iter(loader)

# Build face detector
Dcfg['model']['pretrained'] = None
facedet = Dbuilder.build_model(Dcfg, 0) #args.gpu)
facedet.load_state_dict(torch.load(osp.join('detect', Dcfg['predict']['checkpoint']), map_location=lambda storage, loc: storage))
facedet = facedet.eval().cuda()
Dcfg['transform']['augment']['infer'] = 'load_pipeline'
Dloader = Dbuilder.build_dataloader(Dcfg, ann=[], mode='predict')

# Build classifier
Ccfg['model']['params']['pretrained'] = None
model = Cbuilder.build_model(Ccfg['model']['name'], Ccfg['model']['params'])
model.load_state_dict(torch.load(osp.join('classify', Ccfg['predict']['checkpoint']), map_location=lambda storage, loc: storage))
model = model.eval().cuda()
Cloader = Cbuilder.build_dataloader(Ccfg, data_info={'imgfiles': [], 'labels': []}, train_df=None, train_images=[], mode='predict')

THRESHOLD = 0.55
scores = []
for index in tqdm(range(len(loader)), total=len(loader)):
    try:
        frames = next(loader)
        frames_numpy = np.asarray([_.squeeze(0).numpy() for _ in frames])
        if frames_numpy.shape == (1,):
            scores.append(0.5)
            continue

        f0, f1, f2 = frames[0], frames[10], frames[20]
        r0, r1, r2 = dict(img_info=f0), dict(img_info=f1), dict(img_info=f2)
        Dloader.dataset.pre_pipeline(r0)
        Dloader.dataset.pre_pipeline(r1)
        Dloader.dataset.pre_pipeline(r2)
        r0 = Dloader.dataset.pipeline(r0)
        r1 = Dloader.dataset.pipeline(r1)
        r2 = Dloader.dataset.pipeline(r2)
        boxes = [predict(facedet, r0)[0], predict(facedet, r1)[0], predict(facedet, r2)[0]]
        tmp_bboxes = {
            i : boxes[i][boxes[i][:,-1] >= THRESHOLD]
            for i in [0,1,2]
        }                
        tmp_bboxes = [v[:,:-1] for k,v in tmp_bboxes.items() if v.shape[0] > 0]
        if len(tmp_bboxes) == 0: 
            print ('No faces detected for {} !'.format(videos[index]))
            scores.append(0.5)
            continue
        # No need to scale up
        # Because here we are inputting original frame size images to face detector
        # Enlarge
        tmp_bboxes = [enlarge_box(b, frames_numpy[0].shape, scale=1.1) for b in tmp_bboxes]
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
            new_tmp_bboxes = []
            for box in tmp_bboxes:
                new_tmp_bboxes.append([getCoords(b[:-2]) for b in box])
            tmp_bboxes = np.asarray(new_tmp_bboxes)
            merged_bbox = [[np.min(tmp_bboxes[...,0]), np.max(tmp_bboxes[...,1]), np.min(tmp_bboxes[...,2]), np.max(tmp_bboxes[...,3])]]
        # getCoords returns [x1, x2, y1, x2]
        merged_bbox = [(box[0], box[2], box[1], box[3]) for box in merged_bbox]
        merged_bbox = [intify(box) for box in merged_bbox]
        frames_numpy = np.asarray(frames_numpy)
        faces = [frames_numpy[:,y1:y2,x1:x2,:] for x1,y1,x2,y2 in merged_bbox]
        preds = []
        if Ccfg['dataset']['name'] == 'FaceSeqDataset': 
            N = Cloader.dataset.stack
            N = 2 if N == 'diff' else N
            for face in faces:
                stacklist = [face[i:N+i] for i in range(len(face) - N +1)]
                frlist = []
                for stack in stacklist:
                    stack = Cloader.dataset.process_image(stack)
                    stack = torch.tensor(stack).float().unsqueeze(0)
                    frlist.append(stack)
                frs = torch.cat(frlist, dim=0)
                frs = frs.cuda()
                with torch.no_grad():
                    preds.append(torch.mean(torch.sigmoid(model(frs))).cpu().numpy())
        else:
            for face in faces:
                # face.shape = (N_frames, H, W, C)
                frlist = []
                for fr in face:
                    fr = Cloader.dataset.process_image(fr)
                    fr = torch.tensor(fr).float().unsqueeze(0)
                    frlist.append(fr)
                frs = torch.cat(frlist, dim=0)
                frs = frs.cuda()
                with torch.no_grad():
                    preds.append(torch.mean(torch.sigmoid(model(frs))).cpu().numpy())
        scores.append(np.mean(preds))
    except Exception as e:
       print(e)
       scores.append(0.5)


submission = pd.DataFrame({'filename': [_.split('/')[-1] for _ in videos],
                           'label': scores})
if not osp.exists(osp.dirname(cfg['submission_csv'])):
    os.makedirs(osp.dirname(cfg['submission_csv']))

submission.to_csv(cfg['submission_csv'], index=False)

