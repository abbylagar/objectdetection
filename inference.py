import torch
import numpy as np
import wandb
import labelutils
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import utils
import torchvision
import argparse 
import sys
import math
import time
import datetime
import os
from LitDrinksModel import LitDrinksModel
import torchvision
import cv2
#import detect_utils
import config
import label_utils
from google.colab.patches import cv2_imshow


cur_dir = os.getcwd()
coco_names = config.params['classes']


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Object Detection", add_help=add_help)
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--checkpoint_path", 
    default="/content/drive/MyDrive/Colab Notebooks/object_detection/drinkscheck.ckpt", 
    type=str, help="checkpoint file path") 
    parser.add_argument("--camera",default=0, type=int, help='camera index')
    help_ = "Record video"
    parser.add_argument("--record",
                        default=False,
                        action='store_true', 
                        help=help_)
    parser.add_argument("--output-dir", default=cur_dir, type=str, help="path to save outputs")
    help_ = "Video filename"
    parser.add_argument("--videofilename",
                        default="/content/WIN_20220428_21_12_14_Pro.mp4",
                        help=help_)
    #sys.argv = ['-f']
    args = parser.parse_args()
    return args


# define the torchvision image transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),normalize])


def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    print(pred_scores)
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    pred_scores = pred_scores[pred_scores >= detection_threshold]

    return boxes, pred_classes, outputs[0]['labels'] , pred_scores
    
    
class  VideoDemo():
    def __init__(self,
                 detector,
                 device ,
                 camera=0,
                 width=640,
                 height=480,
                 record=True,
                 filename="/content/WIN_20220428_21_12_14_Pro.mp4"
                 ):
        self.detector  =detector
        self.device =device
        self.camera = camera
        self.detector = detector
        self.width = width
        self.height = height
        self.record = record
        self.filename = filename
        self.videowriter = None
        self.initialize()

    def initialize(self):
        if not self.record:
            self.capture = cv2.VideoCapture(self.camera)
            if not self.capture.isOpened():
                print("Error opening video camera")
                return
    
            # cap.set(cv2.CAP_PROP_FPS, 5)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.record:
            print('self recorded video')
            self.capture = cv2.VideoCapture(self.filename)
            self.videowriter = cv2.VideoWriter("helloworld.mp4",
                                                cv2.VideoWriter_fourcc(*'mp4v'),
                                                20,
                                                (self.width, self.height), 
                                                isColor=True)  




        """"
        if self.record:
          self.capture = cv2.VideoCapture(self.camera)
          if not self.capture.isOpened():
            print("Error opening video camera")
            return

            # cap.set(cv2.CAP_PROP_FPS, 5)
          self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
          self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
          
          
          
          
          
         # if self.record:
            # get the frame width and height
          #frame_width = int(self.capture.get(3))
          #frame_height = int(self.capture.get(4))
          #self.width = frame_width
          #self.height = frame_height
          #video_fps = self.capture.get(cv2.CAP_PROP_FPS)
          self.videowriter = cv2.VideoWriter('/content/drive/MyDrive/Colab Notebooks/finally5_20vps.mp4',
                                              cv2.VideoWriter_fourcc(*'mp4v'),20,
                                              (self.width, self.height), isColor=True)

        if self.record == True:
          print('use recoring')
          self.filename = "/content/WIN_20220428_21_12_14_Pro.mp4"
          self.capture = cv2.VideoCapture(self.filename)
          frame_width = int(self.capture.get(3))
          frame_height = int(self.capture.get(4))
          self.width = frame_width
          self.height = frame_height
          #video_fps = self.capture.get(cv2.CAP_PROP_FPS)
          self.videowriter = cv2.VideoWriter('/content/drive/MyDrive/Colab Notebooks/finally5_20vps.mp4',
                                              cv2.VideoWriter_fourcc(*'mp4v'),20,
                                              (self.width, self.height), isColor=True)

      

        #self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        #self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        """

    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        pos = (10,30)
        font_scale = 0.9
        font_color = (0, 0, 0)
        line_type = 1
        self.capture = cv2.VideoCapture(self.filename)
       
        while(self.capture.isOpened()):
        #while True:
            print('hey')
            start_time = datetime.datetime.now()
            ret, image = self.capture.read()

            #filename = "temp.jpg"
            #cv2.imwrite(filename, image)
            #img = skimage.img_as_float(imread(filename))

            #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #class_names, rects = self.detector.evaluate(image=img)
            
            
            if torch.no_grad:
              boxes, classes, labels ,scores = predict(img, self.detector, self.device, 0.45)

            class_names = classes
            rects = boxes   
            print(class_names)
            #elapsed_time = datetime.datetime.now() - start_time
            #hz = 1.0 / elapsed_time.total_seconds()
            #hz = "%0.2fHz" % hz
            #cv2.putText(image,
            #            hz,
            #            pos,
            #            font,
            #            font_scale,
            #            font_color,
            #            line_type)

            items = {}
            for i in range(len(scores)):
                rect = rects[i]
                x1 = rect[0]
                y1 = rect[1]
                x2 = x1 + rect[2]
                y2 = y1 + rect[3]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                name = class_names[i].split(":")[0]
                if name in items.keys():
                    items[name] += 1
                else:
                    items[name] = 1
                index = label_utils.class2index(name)
                color = label_utils.get_box_rgbcolor(index)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                # print(x1, y1, x2, y2, class_names[i])
                cv2.putText(image,
                            name,
                            (x1, y1-15),
                            font,
                            0.5,
                            color,
                            line_type)
            



            count = len(items.keys())
            if count > 0:
                xmin = 10
                ymin = 10
                xmax = 220
                ymax = 40 + count * 30
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=-1)

                prices = config.params['prices']
                total = 0.0
                for key in items.keys():
                    count = items[key]
                    cost = count * prices[label_utils.class2index(key)]
                    total += cost
                    display = "%0.2f :%dx %s" % (cost, count, key)
                    cv2.putText(image,
                                display,
                                (xmin + 10, ymin + 25),
                                font,
                                0.55,
                                (0, 0, 0),
                                1)
                    ymin += 30

                cv2.line(image, (xmin + 10, ymin), (xmax - 10, ymin), (0,0,0), 1)

                display = "P%0.2f Total" % (total)
                cv2.putText(image,
                            display,
                            (xmin + 5, ymin + 25),
                            font,
                            0.75,
                            (0, 0, 0),
                            1)

            cv2_imshow( image)

            print("writer")
            #cv2.imshow('image', image)
            if self.videowriter is not None:
                if self.videowriter.isOpened():
                    self.videowriter.write(image)
                    print('printing')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()





def main():
    args = get_args_parser() 

    print('entering main')
    print(args.camera)
    print(args.videofilename)
    print(args.device)
    model = LitDrinksModel.load_from_checkpoint(args.checkpoint_path)
    print('printing  device')
    print(args.device)
    model.eval().to(args.device)
    videodemo = VideoDemo(detector = model,device = args.device,
                              camera=args.camera,
                              record=args.record,
                              filename=args.videofilename)
    videodemo.loop()
   
if __name__ == "__main__":
    main()
    
    
    