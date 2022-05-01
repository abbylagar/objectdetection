"""test.py
includes :
-downloading the drinks datasets
-testing the pretrained model

Steps:
1.fork github
2.Change directory
3.!pip install -r requirements.txt
4.!python test.py

Sample can be found at Test_Notebook.ipynb

"""


#import functions
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
import argparse
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
from LitDrinksModel import LitDrinksModel
import tarfile
import gdown






#current working directory
cur_dir =os.getcwd()

config = {
    "num_workers": 2,
    "pin_memory": True,
    "batch_size": 2,
    "dataset": "drinks",
    "train_split": cur_dir +"/datasets/python/drinks/labels_train.csv",
    "test_split": cur_dir +"/datasets/python/drinks/labels_test.csv",
    "num_classes":4
    }


def download_dataset():
    #dataset googledrive link
    url = 'https://drive.google.com/u/0/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
    #output file
    output = 'drinks.tar.gz'
    
    gdown.download(url, output, quiet=False)
    fname = 'drinks.tar.gz'
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path = './datasets/python/')
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()
    print('Done extraction!')



def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Testing", add_help=add_help)
    parser.add_argument("--checkpoint_path", default="./weights/mymodel.ckpt", type=str, help="checkpoint file path")
    parser.add_argument("--data-path", default="/datasets/python/drinks/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="drinks", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    #accelerator
    parser.add_argument("--accelerator", default='gpu')
    #set the checkpoint name
    parser.add_argument("--output-dir", default="./outputs/", type=str, help="path to save outputs")
    
    args = parser.parse_args()
    return args


def main(args):
  
   test_dict, test_classes = labelutils.build_label_dictionary(config['test_split'])
   #train_dict, train_classes =labelutils.build_label_dictionary(config['train_split'])

    #transformation
   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

   transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                        ])

   #train_split = ImageDataset(train_dict, transform)
   test_split = ImageDataset(test_dict, transform)

   #This is approx 95/5 split
   #print("Train split len:", len(train_split))
   print("Test split len:", len(test_split))

   
    #evaluate
   test_loader = DataLoader(test_split,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'],
                      pin_memory=config['pin_memory'],
                      collate_fn=utils.collate_fn
                      )
   print("Testing")  
  #trainer.test(model)
  #evaluate(modelq, test_loader, device='cpu')
  #t#rainer.test(model)


   #load checkpoint
   print(args.checkpoint_path)
   mymodel = LitDrinksModel.load_from_checkpoint(args.checkpoint_path)
   evaluate(mymodel, test_loader, device=args.device)



if __name__ == "__main__":
    args = get_args_parser()
    download_dataset()
    main(args)