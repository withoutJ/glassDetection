import torch
from torch import nn

from models.utils import get_depth_decoder, get_posenet
from models.utils import get_resnet_backbone


class GlassDetectionWithDepth(nn.Module):
    def __init__(self, models):
        super(GlassDetectionWithDepth, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO Finish forward such that outputs contain everything it should
    def forward(self, x):
        outputs = {}
        inputs = x
        return outputs
    

def glass_detection_with_depth(backbone_name, backbone_pretraining, depth_pretraining, 
                               pose_pretraining, freeze_backbone,
                               freeze_glass_decoder, freeze_depth, freeze_pose,
                               frame_ids, num_scales,
                               disable_monodepth, enable_imnet_encoder,
                               disable_pose):
    num_input_frames = len(frame_ids)
    num_pose_frames = num_input_frames

    models = {}

    models["encoder"] = get_resnet_backbone(backbone_name, backbone_pretraining)
    num_ch_enc = models["encoder"].num_ch_enc

    if enable_imnet_encoder:
        models["imnet_encoder"] = get_resnet_backbone(
            backbone_name, 'imnet',
            use_intermediate_layer_getter=False
        )
        for param in models["imnet_encoder"].parameters():
            param.requires_grad = False

    if not disable_pose and not disable_monodepth:
        models.update(get_posenet("resnet18", backbone_pretraining, pose_pretraining, num_pose_frames))
    
    # TODO Add joint depth and glass decoder 

    # TODO Add transformer blocks that uses depth and color to predict glass

    if freeze_backbone:
        print('Freeze backbone weights')
        for param in models["encoder"].parameters():
            param.requires_grad = False

    if not disable_monodepth and freeze_pose:
        print('Freeze pose decoder weights')
        if "pose_encoder" in models:
            for param in models["pose_encoder"].parameters():
                param.requires_grad = False
        for param in models["pose"].parameters():
            param.requires_grad = False
    
    
    model = GlassDetectionWithDepth(models)

    return model