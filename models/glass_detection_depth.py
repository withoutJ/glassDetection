import torch
from torch import nn


class GlassDetectionWithDepth(nn.Module):
    def __init__(self):
        super(GlassDetectionWithDepth, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO Finish forward such that outputs contain everything it should
    def forward(self, x):
        outputs = {}
        inputs = x
        return outputs
    

def glass_detection_with_depth():
    model = GlassDetectionWithDepth()

    return model