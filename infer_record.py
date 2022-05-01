import torch
import argparse
import time
#import detect_utils
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import labelutils
import matplotlib.pyplot as plt
import utils
import torchvision
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
    parser.add_argument("--checkpoint_path", default=cur_dir+"/weights/mymodel.ckpt", type=str, help="checkpoint file path") 
    parser.add_argument("--outputfilename", default='output1', type=str, help="path to save outputs")
    help_ = "Video filename"
    parser.add_argument("--videofilename",
                        default=cur_dir+"/sample/video1.mp4",
                        help=help_)
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
    #print(pred_scores)
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    pred_scores = pred_scores[pred_scores >= detection_threshold]

    return boxes, pred_classes, outputs[0]['labels'] , pred_scores
    



def draw_boxes(boxes, classes, labels, image,scores):
    # read the image with OpenCV
    #image = np.array(image)
    
    class_names = classes
    rects = boxes   
    #print(class_names)
    font = cv2.FONT_HERSHEY_DUPLEX
    pos = (10,30)
    font_scale = 0.9
    font_color = (0, 0, 0)
    line_type = 1
    items = {}
    for i in range(len(scores)):
        rect = rects[i]
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        #x2 = x1 + rect[2]
        #y2 = y1 + rect[3]
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

    #cv2_imshow( image)

    return image




class VideoDector():
  def __init__(self,
                 detector,
                 device,
                 width=640,
                 height=480,
                 videofilename=cur_dir+"/sample/video1.mp4",
                 outputfilename="output1"
                 ):
        self.detector=detector
        self.device =device
        self.width = width
        self.height = height
        self.outputfilename = outputfilename
        self.videofilename = videofilename



        cap = cv2.VideoCapture(self.videofilename)
        if (cap.isOpened() == False):
          print('Error while trying to read video. Please check path again')
        # get the frame width and height
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        save_name = cur_dir +'/outputs/'+self.outputfilename
        # define codec and create VideoWriter object 
        out = cv2.VideoWriter(f"{save_name}.mp4", 
                              cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                              (frame_width, frame_height))


       # model =self.detector

        # read until end of video
        #while(cap.isOpened()):
        while True: 
            # capture each frame of the video
            ret, frame = cap.read()
            #img = cv2.cvtColor( np.float32(frame), cv2.COLOR_BGR2RGB)  
            
           
            if ret == True:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
                # get the start time
                start_time = time.time()
                with torch.no_grad():
                    # get predictions for the current frame
                    boxes, classes, labels ,scores= predict(img,self.detector,self.device, 0.7)
                
                # draw boxes and show current frame on screen
                image =draw_boxes(boxes, classes, labels, frame ,scores)
                # get the end time
                end_time = time.time()
                # get the fps
                fps = 1 / (end_time - start_time)
                # add fps to total fps
              
                # press `q` to exit
                wait_time = max(1, int(fps/4))
                #cv2_imshow(image)
                out.write(image)
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
            else:
                break

        
        # release VideoCapture()
        cap.release()
        # close all frames and video windows
        cv2.destroyAllWindows()
        # calculate and print the average FPS
        filelocation = cur_dir +'/outputs/'+ self.outputfilename
        print(f"Done.Video at"+filelocation)




def main():
    args = get_args_parser() 
    print('converting')
    model = LitDrinksModel.load_from_checkpoint(args.checkpoint_path)
    model.eval().to(args.device)

    VideoDector(detector = model,device = args.device,
                              outputfilename = args.outputfilename,
                              videofilename=args.videofilename)
   
if __name__ == "__main__":
    main()

