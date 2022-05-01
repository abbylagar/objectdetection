"""train.py 

-testing the pretrained model

1.fork  github
2.change directory 
3.!pip install -r requirements.txt
3.create wandb proj. (optional)
4.!python train.py
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


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training", add_help=add_help)
    parser.add_argument("--data-path", default="/datasets/python/drinks/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="drinks", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x     batch_size")
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=2, type=int, metavar="N", help="number of data loading workers           (default: 4)")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    #If different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
    parser.add_argument("--lr",default=0.002,type=float,help="initial learning rate, 0.002 is the default value for         training on 1 gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay")
    parser.add_argument("--norm-weight-decay",
        default=None,type=float,help="weight decay for Normalization layers (default: None, same value as --wd)",)
    parser.add_argument("--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default:             multisteplr)")
    parser.add_argument("--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr        scheduler only)")
    parser.add_argument("--lr-steps",
        default=[16, 22],nargs="+",type=int,help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr       scheduler only)")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    #set output dir = cur_dir
    parser.add_argument("--output-dir", default=cur_dir+"/weights", type=str, help="path to save outputs")
    #set the checkpoint name
    parser.add_argument("--checkpoint_name", default="mymodel", type=str, help="file name of checkpoint")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument("--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of          backbone")
    parser.add_argument("--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip     )")
    parser.add_argument("--sync-bn",dest="sync_bn",help="Use sync batch norm",action="store_true",)
    parser.add_argument("--test-only",dest="test_only",help="Only test the model",action="store_true",)

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default='ResNet50_Weights.IMAGENET1K_V1', type=str, help="the backbone        weights enum name to load")
     
    parser.add_argument("--no-wandb", default=False, action='store_true')
    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    return args



def main(args):
   
   test_dict, test_classes = labelutils.build_label_dictionary(config['test_split'])
   train_dict, train_classes =labelutils.build_label_dictionary(config['train_split'])

    #transformation
   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

   transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                        ])

   train_split = ImageDataset(train_dict, transform)
   test_split = ImageDataset(test_dict, transform)

   #This is approx 95/5 split
   print("Train split len:", len(train_split))
   print("Test split len:", len(test_split))


   model = LitDrinksModel(train_split,test_split ,args,lr=args.lr, batch_size=args.batch_size)
   model.setup()

   #printing the model 
   print(model)

   # wandb
   if args.no_wandb == False :
     wandb_logger = WandbLogger(project="Drinks_detection")
    
   
   
   #checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,filename=args.checkpoint_name)
   """
   checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,filename='{epoch}-'+args.checkpoint_name,
   every_n_epochs =5)
   trainer = Trainer(callbacks=[checkpoint_callback])
   """    
   trainer = Trainer(
      #change accelarator to cpu
      accelerator=args.accelerator,
      devices=args.devices,
      max_epochs=args.epochs,
      logger=wandb_logger if not args.no_wandb else None,
      enable_checkpointing=False
      #callbacks=[checkpoint_callback]
      )

   start_time = time.time()

   trainer.fit(model)
   
   #save checkpoint
   chkpath =cur_dir+"/weights/" +args.checkpoint_name +".ckpt"
   trainer.save_checkpoint(filepath = chkpath )
   
   final_time = time.time()
   total_time = final_time - start_time
   total_time_str = str(datetime.timedelta(seconds=int(total_time)))
   
   print(f"Training time {total_time_str}")
  
  
   
   #if args.no_wandb == False :
   wandb.finish()
   

   
   print("Training metrics")  
  #trainer.test(model)
  #evaluate(modelq, test_loader, device='cpu')

    
    
    #evaluate
   train_loader = DataLoader(train_split,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'],
                      pin_memory=config['pin_memory'],
                      collate_fn=utils.collate_fn
                      )
   #load checkpoint
   mymodel = LitDrinksModel.load_from_checkpoint(chkpath)
   evaluate(mymodel, train_loader, device=args.device)

                  

   """
   #evaluate
   print('test metrics')
   test_loader = DataLoader(test_split,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config['num_workers'],
                      pin_memory=config['pin_memory'],
                      collate_fn=utils.collate_fn
                      )
   #load checkpoint"
   #mymodel = LitDrinksModel.load_from_checkpoint(chkpath)
   evaluate(mymodel, test_loader, device=args.device)
   """
                  

if __name__ == "__main__":
    args = get_args_parser()
    main(args)