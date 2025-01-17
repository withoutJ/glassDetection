from loss.monodepth_loss import MonodepthLoss
from loss.losses import structure_loss

def get_glass_detection_loss(opt):
    return structure_loss

def get_monodepth_loss(cfg, is_train, batch_size=None):
    if batch_size is None:
        batch_size = cfg["training"]["batch_size"]
    return MonodepthLoss(**cfg["training"]["monodepth_loss"],
                         batch_size=batch_size,
                         is_train=is_train)