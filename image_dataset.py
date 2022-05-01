"""ImageDataset
-4/2022 alagar Dataloader functions
"""

import torch
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
        # retrieve all bounding boxes
        boxes = self.dictionary[key]
        # open the file as a PIL image
        img = Image.open(key)
        
        #boxes
        boxes_data=[]
        for i in range(len(boxes)):
          dimen = boxes[i]
          #boxes_data.append(dimen[:-1])
          xmin = dimen[0]
          xmax = dimen[1]
          ymin = dimen[2]
          ymax = dimen[3]
          boxes_data.append([xmin, ymin, xmax, ymax])
      
         # convert everything into a torch.Tensor
        boxes_data = torch.as_tensor(boxes_data, dtype=torch.float32)

        #labels
        labels=[]
        for i in range(len(boxes)):
          dimen = boxes[i]
          dlabel = dimen[-1:].item()
          labels.append(dlabel)
        
        labels =torch.as_tensor(labels, dtype=torch.int64)


        area = (boxes_data[:, 3] - boxes_data[:, 1]) * (boxes_data[:, 2] - boxes_data[:, 0])
        
          # suppose all instances are not crowd
        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_data
        target["labels"] = labels
        #target["masks"] = masks
        key = torch.as_tensor(idx,dtype=torch.int64)
        target["image_id"] = key
        target["area"] = area
        target["iscrowd"] = iscrowd

    
        # apply the necessary transforms
        # transforms like crop, resize, normalize, etc
        if self.transform:
            img = self.transform(img)

        # return a list of images and corresponding labels
        #return img, boxes
        return img, target


