"""Pytorch Lightning Module
"""

import torch
import numpy as np
import wandb
import labelutils
from image_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import utils
import torchvision
from argparse import ArgumentParser
from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
import sys
import math
import time
import datetime
import os
from engine import train_one_epoch, evaluate


config = {
    "num_workers": 2,
    "pin_memory": True,
    "batch_size": 2,
    "dataset": "drinks",
    "train_split": "datasets/python/drinks/labels_train.csv",
    "test_split": "datasets/python/drinks/labels_test.csv",
    "num_classes":4
    
}


class LitDrinksModel(LightningModule ):
  #modified 
  #batch_size = from 32 to 10
  #num_classes = 4 
    def __init__(self, train_split, test_split ,args ,num_classes=config['num_classes'], lr=0.002, batch_size=config['batch_size']):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
        self.train_split = train_split
        self.test_split  = test_split
        self.args =args
       
    def forward(self, x):
        self.model.to(self.args.device)
        return self.model(x)

    def configure_optimizers(self):
        if self.args.norm_weight_decay is None:
          parameters = [p for p in self.model.parameters() if p.requires_grad]
        else:
          param_groups = torchvision.ops._utils.split_normalization_params(self.model)
          wd_groups = [self.args.norm_weight_decay, self.args.weight_decay]
          parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
        
        opt_name = self.args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(parameters,lr=self.args.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay,
                nesterov="nesterov" in opt_name)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {self.args.opt}. Only SGD and AdamW are supported.")

        #return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        #params = [p for p in self.model.parameters() if p.requires_grad]
        #optimizer =torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        self.args.lr_scheduler = self.args.lr_scheduler.lower()
        if self.args.lr_scheduler == "multisteplr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_steps, gamma=self.args.lr_gamma)
        elif self.args.lr_scheduler == "cosineannealinglr":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        else:
            raise RuntimeError(f"Invalid lr scheduler '{self.args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported.")
        return ([optimizer],[lr_scheduler])
        

    # this is called after model instatiation to initiliaze the datasets and dataloaders
    def setup(self, stage=None):
        self.train_dataloader()
        self.test_dataloader()

    # build train and test dataloaders using MNIST dataset
    # we use simple ToTensor transform
    def train_dataloader(self):
        train_loader = DataLoader(self.train_split,
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=config['num_workers'],
                          pin_memory=config['pin_memory'],
                          collate_fn=utils.collate_fn
                          )
        return train_loader


    def test_dataloader(self):
        test_loader = DataLoader(self.test_split,
                         batch_size=config['batch_size'],
                         shuffle=False,
                         num_workers=config['num_workers'],
                         pin_memory=config['pin_memory'],
                         collate_fn=utils.collate_fn
                         )
        return test_loader


    # this is called during fit()
    def training_step(self, batch, batch_idx):
        images, targets = batch
       
        #for images, targets in metric_logger.log_every(batch, args.print_freq, header):
        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device =self.args.device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #with torch.cuda.amp.autocast(enabled=scaler is not None):
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        
        #return metric_logger
        self.log_dict(loss_dict_reduced)
        self.log("step train_loss", losses_reduced)
        return {"loss": losses_reduced}



    # calls to self.log() are recorded in wandb
    def training_epoch_end(self, outputs):
      avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
      self.log("train_loss", avg_loss, on_epoch=True)
      self.log("train_loss", avg_loss, on_epoch=True)



