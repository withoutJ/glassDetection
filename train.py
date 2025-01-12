import argparse
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
import time
import psutil

from loader.dataloader import ImageDataset
from utils.utils import get_logger, save_checkpoint, load_checkpoint
from utils.optimizers import get_optimizer
from evaluation.metrics import runningScore, AverageMeter, AverageMeterDict
from models import get_model


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Trainer:
    def __init__(self, opt, writer, logger):
    
        torch.backends.cudnn.benchmark = True
        self.opt = opt
        self.writer = writer
        self.logger = logger
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "cpu")
        
        # setup training
        self.start_epoch = 0
        self.start_n_iter = 0
        
        # TODO setup the right model
        self.model = get_model(opt["model"]).to(self.device)

        self.optim = get_optimizer(self.opt)

        # TODO setup dataloader correctly
        self.train_dataloader = 0

        self.scaler = torch.GradScaler('cuda', enabled = self.opt['training']['amp'])

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