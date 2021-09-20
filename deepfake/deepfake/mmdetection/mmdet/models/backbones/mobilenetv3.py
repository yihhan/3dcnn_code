import torch.nn as nn

from ..timm.models import mobilenetv3_large_100
from ..registry import BACKBONES


@BACKBONES.register_module
class MobileNetV3(nn.Module):

    def __init__(self, pretrained=True, norm_eval=False):
        super().__init__()
        self.net = mobilenetv3_large_100(pretrained=pretrained, features_only=True)
        self.norm_eval = norm_eval

    def forward(self, x):
        return self.net(x)[::-1][1:]

    def init_weights(self, **kwargs):
        pass

    def train(self, mode=True):
        super().train(mode)
        #self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def eval(self):
        self.train(mode=False)

