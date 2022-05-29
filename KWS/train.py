
from torch import nn
import torch
import torchaudio ,torchvision
import os
import matplotlib.pyplot as plt 
import librosa
import argparse
import numpy as np
import wandb
from pytorch_lightning import LightningModule, Trainer, LightningDataModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torchvision.transforms import ToTensor
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.speechcommands import load_speechcommands_item
from argparse import ArgumentParser
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import accuracy
from einops import rearrange





from dataloader_module import LitKWS
from lit_transformer import LitTransformer


def get_args():
    parser = ArgumentParser(description='PyTorch Transformer')
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--embed_dim', type=int, default=80, help='embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='num_heads')

    parser.add_argument('--patch_num', type=int, default=32, help='patch_num')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--max-epochs', type=int, default=35, metavar='N',
                        help='number of epochs to train (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0)')

    parser.add_argument('--accelerator', default='gpu', type=str, metavar='N')
    parser.add_argument('--devices', default=1, type=int, metavar='N')
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='N')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N')

    parser.add_argument("--no-wandb", default=False, action='store_true')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_args()
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
             'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
   
    # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


    model_checkpoint = ModelCheckpoint(
    dirpath=os.path.join("./checkpoints"),
    filename="kws_best_acc",
    save_top_k=1,
    verbose=True,
    monitor='test_acc',
    mode='max',
    )

    if args.no_wandb == False :
     wandb_logger = WandbLogger(project="KWS")


    # define a metric we are interested in the minimum of
    #wandb.define_metric("test_loss", summary="min")
    # define a metric we are interested in the maximum of
    #wandb.define_metric("test_acc", summary="max")

    datamodule = LitKWS(
        class_dict=CLASS_TO_IDX , 
        batch_size=args.batch_size,
                        patch_num=args.patch_num, 
                        num_workers=args.num_workers * args.devices
          )
    datamodule.prepare_data()

    data = iter(datamodule.train_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]
    print("Embed dim:", args.embed_dim)
    print("Patch size:", 64 // args.patch_num)
    print("Sequence length:", seqlen)


    model = LitTransformer(num_classes=37, lr=args.lr, epochs=args.max_epochs, 
                           depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                           patch_dim=patch_dim, seqlen=seqlen,)

    trainer = Trainer(accelerator=args.accelerator, devices=args.devices,
                      max_epochs=args.max_epochs, precision=16 if args.accelerator == 'gpu' else 32,
                       logger=wandb_logger if not args.no_wandb else None,
                       callbacks=[model_checkpoint])
                    
           
    wandb.define_metric('test_acc',summary='max')
    trainer.fit(model, datamodule=datamodule)

    wandb.finish()