from torch.utils.data import Dataset, Sampler

import torch
import pandas as pd
import numpy as np
import time
import cv2


class FaceDataset(Dataset):

    def __init__(self, 
                 imgfiles, 
                 labels, 
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor):

        self.imgfiles = imgfiles
        self.labels = labels
        self.videos = [_.split('/')[-2] for _ in imgfiles]
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.imgfiles)

    def process_image(self, img):
        X = self.pad(img) 
        X = self.resize(image=X)['image'] 
        if self.crop: X = self.crop(image=X)['image']
        if self.transform: X = self.transform(image=X)['image'] 
        X = self.preprocessor.preprocess(X) 
        return X.transpose(2, 0, 1)

    def __getitem__(self, i):
        X = cv2.imread(self.imgfiles[i])
        while X is None:
            print('Failed to read {}'.format(self.imgfiles[i]))
            i = np.random.choice(range(len(self.imgfiles)))
            X = cv2.imread(self.imgfiles[i])
        X = self.process_image(X)
        y = self.labels[i]
        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        # TODO: To use .cuda in Dataset, need to set_start_method('spawn')
        # Thus may be better to move .cuda to Trainer class
        return X, y


class FaceSeqDataset(Dataset):

    def __init__(self, 
                 imgfiles, 
                 labels, 
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor,
                 stack=None,
                 grayscale=False,
                 vstack=True,
                 dim=2):

        self.imgfiles = imgfiles
        self.labels = labels
        self.videos = [_[0].split('/')[-2] for _ in imgfiles]
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor
        self.stack = stack
        self.grayscale = grayscale
        self.vstack = vstack
        self.dim = dim

    def __len__(self):
        return len(self.imgfiles)

    def read_seq(self, i):
        X = []
        for imgfi in self.imgfiles[i]:
            img = np.expand_dims(cv2.imread(imgfi, 0), axis=-1) if self.grayscale else cv2.imread(imgfi)
            if img is None:
                print('Failed to read {}'.format(imgfi))
                return None
            else:
                X.append(img)
        return X

    def get_stack(self, i):
        X = self.read_seq(i)
        while type(X) == type(None):
            i = np.random.choice(range(len(self.imgfiles)))
            X = self.read_seq(i)
        return X

    def process_image(self, img):
        X = [self.pad(_) for _ in img] 
        X = [self.resize(image=_)['image'] for _ in X]
        if self.crop: X = [self.crop(image=X)['image'] for _ in X]
        if self.transform: 
            to_transform = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_transform.update({'image': X[0]})
            transformed = self.transform(**to_transform)
            X = [transformed['image']] + [transformed['image{}'.format(_)] for _ in range(1,len(X))]
        X = [self.preprocessor.preprocess(_) for _ in X]
        X = [_.transpose(2, 0, 1) for _ in X] 
        X = np.asarray(X)
        # X.shape = (N, C, H, W)
        if self.dim == 3:
            X = X.transpose(1, 0, 2, 3)
            # X.shape = (C, N, H, W)
        if self.stack and self.stack != 'diff':
            if self.vstack:
                X = np.vstack(X)
                # X.shape = (N*C, H, W)
        return X

    def __getitem__(self, i):
        X = self.get_stack(i)
        X = self.process_image(X)
        y = self.labels[i]
        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        # TODO: To use .cuda in Dataset, need to set_start_method('spawn')
        # Thus may be better to move .cuda to Trainer class
        return X, y


class PartSampler(Sampler):

    def __init__(self,
        dataset,
        train_df):
        super().__init__(data_source=dataset)
        self.manager = {}
        for i in train_df['folder'].unique():
            self.manager[i] = {}
            part_df = train_df[train_df['folder'] == i]
            for real_video, _df in part_df.groupby('original'):
                self.manager[i][real_video] = list(np.unique(_df['filename'][_df['label'] == 1]))

        # Dict where key=video and value=list of indices corresponding
        # to face images for that video
        self.indices_map = {vid : list(_df.index) for vid, _df in train_df.groupby('filename')}
        self.length = len(train_df['original'].unique())*2

    def __iter__(self):
        all_indices = []
        parts = [*self.manager]
        while len(all_indices) < self.length:
            # Shuffle parts
            parts = np.random.permutation(parts)
            # Get reals from the first half
            # Fakes from the second half
            shuf1 = parts[:len(parts)//2]
            shuf2 = parts[len(parts)//2:]
            # Now, sample a real video from shuf1
            real_batch = []
            for i in shuf1:
                real_batch.append(np.random.choice([*self.manager[i]]))
            # Sample a fake video from shuf2
            # Avoid sampling a fake video from a real video sampled above
            fake_batch = []
            for i in shuf2:
                sampled_real = np.random.choice([*self.manager[i]])
                while sampled_real in real_batch:
                    sampled_real = np.random.choice([*self.manager[i]])
                fake_batch.append(np.random.choice(self.manager[i][sampled_real]))
            # Now, we need to map these videos to indices of the face images
            real_indices = [np.random.choice(self.indices_map[i]) for i in real_batch]
            fake_indices = [np.random.choice(self.indices_map[i]) for i in fake_batch]
            # Merge alternating
            indices = real_indices + fake_indices
            # List assigned to [::2] needs to be the longer one
            # So try-except since the max length difference will always be 1
            try:
                indices[::2]  = real_indices
                indices[1::2] = fake_indices
            except ValueError:
                indices[::2]  = fake_indices
                indices[1::2] = real_indices
            # Permute
            indices = np.random.permutation(indices)
            all_indices.extend(indices)
        return iter(all_indices)

    def __len__(self):
        return self.length


class PartSeqSampler(Sampler):

    def __init__(self,
        dataset,
        train_df,
        train_images):
        super().__init__(data_source=dataset)
        self.manager = {}
        videos = np.asarray([im[0].split('/')[-2] for im in train_images])
        for i in train_df['folder'].unique():
            self.manager[i] = {}
            part_df = train_df[train_df['folder'] == i]
            for real_video, _df in part_df.groupby('original'):
                self.manager[i][real_video] = list(np.unique(_df['filename'][_df['label'] == 1]))

        # Dict where key=video and value=list of indices corresponding
        # to face images for that video
        self.indices_map = {vid+'.mp4' : list(np.where(videos == vid)[0]) for vid in np.unique(videos)}
        self.length = len(train_df['original'].unique())*2

    def __iter__(self):
        all_indices = []
        parts = [*self.manager]
        while len(all_indices) < self.length:
            # Shuffle parts
            parts = np.random.permutation(parts)
            # Get reals from the first half
            # Fakes from the second half
            shuf1 = parts[:len(parts)//2]
            shuf2 = parts[len(parts)//2:]
            # Now, sample a real video from shuf1
            real_batch = []
            for i in shuf1:
                real_batch.append(np.random.choice([*self.manager[i]]))
            # Sample a fake video from shuf2
            # Avoid sampling a fake video from a real video sampled above
            fake_batch = []
            for i in shuf2:
                sampled_real = np.random.choice([*self.manager[i]])
                while sampled_real in real_batch:
                    sampled_real = np.random.choice([*self.manager[i]])
                fake_batch.append(np.random.choice(self.manager[i][sampled_real]))
            # Now, we need to map these videos to indices of the face images
            real_indices = [np.random.choice(self.indices_map[i]) for i in real_batch]
            fake_indices = [np.random.choice(self.indices_map[i]) for i in fake_batch]
            # Merge alternating
            indices = real_indices + fake_indices
            # List assigned to [::2] needs to be the longer one
            # So try-except since the max length difference will always be 1
            try:
                indices[::2]  = real_indices
                indices[1::2] = fake_indices
            except ValueError:
                indices[::2]  = fake_indices
                indices[1::2] = real_indices
            # Permute
            indices = np.random.permutation(indices)
            all_indices.extend(indices)
        return iter(all_indices)

    def __len__(self):
        return self.length







