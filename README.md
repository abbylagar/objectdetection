# Drinks Object Detection using Faster Rcnn Resnet50 FPN
> created by : Abegail Lagar

Tools used:
```
Pytorch Lightning
Wandb
```
### Folder Structure
    .
    ├── weights                   # Pretrained weights
    ├── outputs                   # Annotated output
    ├── samples                   # Raw video
    ├── datasets                  # Dataset folder
    │   └── python          
    │         └── drinks          # Drinks Dataset
    ├── various codes             #train ,test codes and other utilities 
    └── README.md
    

### Set-up procedures:
steps before training and testing the model

### 1. Clone the repository:
```
! git clone link/to/your/repo
```

### 2. Go to your project directory
```
%cd <project_folder>
```

### 3. Install the requirements
```
!pip install -r requirements.txt
```

### Testing the model
Sample run in the Drinks_detection.ipynb
```
!python3 test.py
```
Output metrics:


![image](https://user-images.githubusercontent.com/67377766/166150634-35123488-34a6-481f-909a-76b7d593b449.png)

### Training the model
Sample run in the Drinks_detection.ipynb .  
If wandb visualization prompts , enter 3 (No visualization)
```
!python3 train.py
```
Output metrics after 50 epochs

![image](https://user-images.githubusercontent.com/67377766/166149359-fd1f2af6-2444-4f56-ac29-5583a53ee5c3.png)


### Sample Video 
Sample videos with object detection tracking is in the outputs folder Filename : output1.mp4
```
!python3 infer_record.py 
```


![image](https://user-images.githubusercontent.com/67377766/166147798-7cacd8d7-eb02-45e4-834d-11a782d92170.png)


### References:
 1. [TorchVision Object Detection Finetuning Tutoria](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
 2. [Pytorch Vision Reference](https://github.com/pytorch/vision/tree/main/references/detection)
 3. [Object Detection Using Colab webcam](https://github.com/TannerGilbert/Tensorflow-Object-Detection-with-Tensorflow-2.0/blob/master/live_object_detection.ipynb)

