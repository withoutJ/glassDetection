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

from loader.dataloader import ImageDataset
from utils.utils import get_logger, save_checkpoint, load_checkpoint
from utils.optimizers import get_optimizer
from evaluation.metrics import runningScore, AverageMeter, AverageMeterDict


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
        
        # TODO setup the right model
        self.model = 0 

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
            start_epoch = ckpt['epoch']+1
            start_n_iter = ckpt['n_iter']
            self.best_iou = ckpt["best_iou"]
            self.optim.load_state_dict(ckpt['optim'])
            self.logger.info("Last checkpoint restored")
            self.logger.info(
                "Loaded checkpoint '{}' (iter {})".format(self.opt["training"]["resume"], start_epoch)
            )
        else:
            self.logger.info("No checkpoint found at '{}'".format(self.opt["training"]["resume"]))
    
    def train_step(inputs):
        pass

    
    def train(self):
        start_n_iter = 0
        start_epoch = 0
        if self.opt["training"]["resume"] is not None:
            self.load_resume()

        train_loss_meter = AverageMeterDict()
        time_meter = AverageMeter()

        for epoch in range(start_epoch, opt["training"]["epochs"]):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            start_time = time.time()

            for i, inputs in pbar:
                loss = self.train_step(inputs)

                time_meter.update(time.time() - start_time)
                train_loss_meter.update(loss)
                
                
                start_time = time.time()

    def validate(self, step):
        pass

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