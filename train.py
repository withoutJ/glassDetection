import argparse
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
import time
import psutil
from copy import deepcopy

from loader.dataloader import ImageDataset
from utils.utils import get_logger, save_checkpoint, load_checkpoint
from utils.optimizers import get_optimizer
from evaluation.metrics import runningScore, AverageMeter, AverageMeterDict
from models import get_model
from loader import build_dataset
from loss import get_glass_detection_loss, get_monodepth_loss
from utils.schedulers import get_scheduler

def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# TODO Return dictioniary with all trainig params
def get_train_params(model, opt):
    pass

class Trainer:
    def __init__(self, opt, writer, logger):
    
        torch.backends.cudnn.benchmark = True
        self.opt = opt
        self.writer = writer
        self.logger = logger
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "cpu")
        
        self.IoU = 0
        
        # setup training
        self.start_epoch = 0
        self.start_n_iter = 0
        
        # TODO setup the right model
        self.model = get_model(opt["model"]).to(self.device)

        # Setup optimizer
        optimizer_cls = get_optimizer(self.opt)
        optimizer_params = {k: v for k, v in self.opt["training"]["optimizer"].items() if
                            k not in ["name", "backbone_lr", "pose_lr", "depth_lr", "segmentation_lr"]}
        train_params = get_train_params(self.model, self.opt)
        self.optimizer = optimizer_cls(train_params, **optimizer_params)

        self.scheduler = get_scheduler(self.optimizer, self.opt["training"]["lr_schedule"])

        # TODO setup dataloader correctly
        data_opt = deepcopy(self.opt["data"])
        self.train_dataset = build_dataset(data_opt, split="train")
        self.test_dataset = build_dataset(data_opt, split="test")

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.opt["training"]["batch_size"],
            num_workers=self.opt["training"]["num_workers"],
            shuffle=self.opt["data"]["shuffle_trainset"],
            pin_memory=True,
            drop_last=True)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.opt["training"]["val_batch_size"],
            num_workers=self.opt["training"]["n_workers"],
            pin_memory=True)

        self.scaler = torch.GradScaler('cuda', enabled = self.opt["training"]["amp"])

        # Setup losses
        self.loss_fn = get_glass_detection_loss(self.opt)
        self.monodepth_loss_calculator_train = get_monodepth_loss(self.opt, is_train=True)
        self.monodepth_loss_calculator_val = get_monodepth_loss(self.opt, is_train=False, batch_size=self.val_batch_size)

        # TODO Add early stopping
        self.earlyStopping = None

        setup_seeds(opt.get("seed", 42))
    
    def load_resume(self):
        if os.path.isfile(self.cfg["training"]["resume"]):
            self.logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(self.opt["training"]["resume"])
            )
            ckpt = load_checkpoint(self.opt["resume"])
            self.model.load_state_dict(ckpt['net'])
            self.start_epoch = ckpt['epoch']+1
            self.start_n_iter = ckpt['n_iter']
            self.best_iou = ckpt["best_iou"]
            self.optim.load_state_dict(ckpt['optim'])
            self.logger.info("Last checkpoint restored")
            self.logger.info(
                "Loaded checkpoint '{}' (iter {})".format(self.opt["training"]["resume"], self.start_epoch)
            )
        else:
            self.logger.info("No checkpoint found at '{}'".format(self.opt["training"]["resume"]))
    
    def train_step(self, inputs, step):
        self.model.train()

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()


        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(inputs)

        # TODO turn off trainig when freezing the backbone

        # Train monodepth
        if self.cfg["training"]["monodepth_lambda"] > 0:
            for k, v in outputs.items():
                if "depth" in k or "cam_T_cam" in k:
                    outputs[k] = v.to(torch.float32)
            self.monodepth_loss_calculator_train.generate_images_pred(inputs, outputs)
            mono_losses = self.monodepth_loss_calculator_train.compute_losses(inputs, outputs)
            mono_lambda = self.opt["training"]["monodepth_lambda"]
            mono_loss = mono_lambda * mono_losses["loss"]
            feat_dist_lambda = self.opt["training"]["feat_dist_lambda"]
            if feat_dist_lambda > 0:
                feat_dist = torch.dist(outputs["encoder_features"], outputs["imnet_features"], p=2)
                feat_dist_loss = feat_dist_lambda * feat_dist
            mono_total_loss = mono_loss + feat_dist_loss

            self.scaler.scale(mono_total_loss).backward(retain_graph=True)


        # TODO Train glass detection


        self.scaler.step(self.optimizer)
        self.scaler.update()
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics=self.IoU)
        else:
            self.scheduler.step()

        # TODO return all needed losses
        return {}
    
    def train(self):
        self.start_n_iter = 0
        self.start_epoch = 0
        if self.opt["training"]["resume"] is not None:
            self.load_resume()

        train_loss_meter = AverageMeterDict()
        time_meter = AverageMeter()

        for epoch in range(self.start_epoch, opt["training"]["epochs"]):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for i, inputs in pbar:
                step = epoch*len(pbar) + i
                start_time = time.time()

                loss = self.train_step(inputs, step)

                time_meter.update(time.time() - start_time)
                train_loss_meter.update(loss)
                

                
                if (step+1) % self.opt["trainig"]["print_interval"] == 0:
                    progress_str = (
                        f'Epoch: {epoch}/{self.opt["training"]["epochs"]}], '
                        f'Loss: {train_loss_meter.avgs["total_loss"]:.4f}, '
                        f'Time/Image: {time_meter.avg / self.cfg["training"]["batch_size"]:.4f}'
                    )
                    pbar.set_description(progress_str)
                    pbar.refresh()
                    self.logger.info(progress_str)

                    for k, v in train_loss_meter.avgs.items():
                        self.writer.add_scalar("training/" + k, v, step + 1)
                    self.writer.add_scalar("training/time_per_image", time_meter.avg / self.opt["training"]["batch_size"], step + 1)
                    self.writer.add_scalar("training/memory", psutil.virtual_memory().used / 1e9, step + 1)
                    time_meter.reset()
                    train_loss_meter.reset()
                
                # TODO validation

                        

    def validate(self, step):
        self.model.eval()


def main(opt):
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    if "name" not in opt:
        opt["name"] = "run" + run_id
    opt['training']['log_path'] = os.path.join( opt['training']['log_path'], opt['name']) 
    print('Start', opt['name'])

    logdir = opt['training']['log_path']

    writer = SummaryWriter(log_dir=logdir, filename_suffix='.metrics')

    print("RUNDIR: {}".format(logdir))
    with open(logdir + "/cfg.yml", 'w') as fp:
        yaml.dump(opt, fp)

    logger = get_logger(logdir)
    logger.info("Begin")

    trainer = Trainer(opt, writer, logger)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/multiview.yml",
        help="Configuration file to use",
    )
    
    args = parser.parse_args()
    with open(args.config) as fp:
        opt = yaml.safe_load(fp)

    main(opt)