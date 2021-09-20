import logging
import scipy.misc
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .backbones import *


class SingleHead(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3):

        super(SingleHead, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)[:,0]

class SingleHead3D(SingleHead): pass

class FeatureComb(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 combine='conv',
                 num_stack=21):

        super(FeatureComb, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        if combine == 'conv':
            self.combine = nn.Conv1d(num_stack, 1, kernel_size=1, stride=1, bias=False)
        elif combine == 'max':
            self.combine = 'max'
        elif combine == 'avg':
            self.combine = 'avg'
        self.fc = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        # x.shape = (B, N, C, H, W)
        # B = batch size
        # N = number of images
        # (C, H, W) = image
        # Get features over N
        feats = []
        for _ in range(x.size(1)):
            feats.append(self.dropout(self.backbone(x[:,_])).unsqueeze(1))
        feats = torch.cat(feats, dim=1)
        # feats.shape = (B, N, dim_feats)
        if isinstance(self.combine, nn.Module):
            combined = self.combine(feats)[:,0]
        elif self.combine == 'max':
            combined = torch.max(feats, dim=1)[0]
        elif self.combine == 'avg':
            combined = torch.mean(feats, dim=1)
        return self.fc(combined)[:,0]


class DiffModel(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 diff_only=False):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(int(2*dim_feats) if diff_only else int(4*dim_feats), num_classes)
        self.diff_only = diff_only

    def forward_train(self, x):
        # x.shape = (B, 2, C, H, W)
        # B = batch size
        # (C, H, W) = image
        # Get features over N
        assert x.size(1) == 2
        feat1 = self.backbone(x[:,0])
        feat2 = self.backbone(x[:,1])
        l1 = torch.abs(feat1-feat2)
        l2 = l1 ** 2
        if self.diff_only:
            feats = torch.cat((l1, l2), dim=1)
        else:
            ad = feat1+feat2
            mu = feat1*feat2
            feats = torch.cat((l1, l2, ad, mu), dim=1)
        feats = self.dropout(feats)
        return self.fc(feats)[:,0]

    def forward_test(self, x):
        # x.shape = (B, N, C, H, W)
        out = []
        for i in range(0, x.size(1), 2):
            out.append(self.forward_train(x[:,i:i+2]))
        return torch.mean(torch.stack(out, dim=0), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)



class MultiDiffModel(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 diff_only=False,
                 num_stack=5):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.combos = list(itertools.combinations(range(num_stack), 2))
        self.combine = nn.Conv1d(len(self.combos), 1, kernel_size=1, stride=1, bias=False)
        self.fc = nn.Linear(int(2*dim_feats) if diff_only else int(4*dim_feats), num_classes)
        self.diff_only = diff_only

    def get_feature(self, x, y):
        l1 = torch.abs(x-y)
        l2 = l1 ** 2
        if self.diff_only:
            feats = torch.cat((l1, l2), dim=1)
        else:
            ad = x+y
            mu = x*y
            feats = torch.cat((l1, l2, ad, mu), dim=1)
        return feats

    def forward(self, x):
        # x.shape = (B, N, C, H, W)
        # B = batch size
        # N = number of images
        # (C, H, W) = image
        # Get features over N
        feats = []
        for i in range(x.size(1)):
            feats.append(self.backbone(x[:,i]).unsqueeze(1))
        feats = torch.cat(feats, dim=1)
        # Compute versus for all features
        bigfeats = []
        for c in self.combos:
            bigfeats.append(self.get_feature(feats[:,c[0]], feats[:,c[1]]).unsqueeze(1))
        del feats
        bigfeats = torch.cat(bigfeats, dim=1)
        bigfeats = self.combine(bigfeats)
        bigfeats = self.dropout(bigfeats)
        return self.fc(bigfeats)[:,0,0]


class RNNHead(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 rnn='gru',
                 num_stack=21,
                 hidden_size=None,
                 bidirectional=True,
                 num_layers=2,
                 gpu=True):

        super(RNNHead, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        if rnn.lower() == 'gru':
            rnn_module = nn.GRU
        elif rnn.lower() == 'lstm':
            rnn_module = nn.LSTM
        else:
            raise Exception('`rnn` must be one of [`GRU`, `LSTM`]')

        if hidden_size is None:
            if bidirectional:
                hidden_size = dim_feats // 2
            else:
                hidden_size = dim_feats
        
        self.gpu = gpu
        self.bidirectional = bidirectional
        self.num_layers = num_layers   
        self.hidden_size = hidden_size

        self.rnn = rnn_module(input_size=dim_feats, 
                              hidden_size=hidden_size, 
                              num_layers=num_layers, 
                              bidirectional=bidirectional,
                              batch_first=True)
        self.conv = nn.Conv1d(21, 1, kernel_size=1, stride=1, bias=False)
        self.final_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, num_classes)

    def init_hidden(self, batch_size): 
        hidden_a = Variable(torch.randn(self.num_layers*2 if self.bidirectional else self.num_layers, batch_size, self.hidden_size))
        hidden_b = Variable(torch.randn(self.num_layers*2 if self.bidirectional else self.num_layers, batch_size, self.hidden_size))
        if self.gpu: 
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()
        if isinstance(self.rnn, nn.GRU):
            return hidden_a
        elif isinstance(self.rnn, nn.LSTM):
            return (hidden_a, hidden_b)

    def forward(self, x):
        # x.shape = (B, N, C, H, W)
        # B = batch size
        # N = number of images
        # (C, H, W) = image
        # Get features over N
        feats = []
        for _ in range(x.size(1)):
            feats.append(self.dropout(self.backbone(x[:,_])).unsqueeze(1))
        feats = torch.cat(feats, dim=1)
        # feats.shape = (B, N, dim_feats)
        self.hidden = self.init_hidden(x.size(0))
        out, self.hidden = self.rnn(feats, self.hidden)
        out = self.conv(out)[:,0]
        out = self.final_dropout(out)
        return self.fc(out)[:,0]












