import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from factory.models.backbones import resnet50
from factory.models import RNNHead

X = torch.from_numpy(np.ones((8,21,3,128,128))).float()

model = RNNHead(backbone='resnet34', pretrained=None, num_classes=1, dropout=0.2, gpu=False)

yhat = model(X)

backbone, _ = resnet50(pretrained=None)
feats = []
for _ in range(X.size(1)):
    feats.append(backbone(X[:,_,...]))

feats = torch.stack(feats, dim=1)

lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=2, batch_first=True, bidirectional=True)
conv = nn.Conv1d(21, 1, kernel_size=1, stride=1, bias=False)


