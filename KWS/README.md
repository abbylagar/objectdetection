# Key Word Spotting 
> created by : Abegail Lagar

Tools used:
```
Pytorch Lightning
Wandb
tkinter
PyAudio
torchaudio
```
### Folder Structure
    .
    ├── checkpoints               # Pretrained weights
    ├── outputs                   # recorded voice folder
    ├── speechcommands            # Dataset folder
    ├── train.py                  # training code
    ├── infer.py                  # run keyword spotting app
    └── README.md
    

### Set-up procedures:
steps before training and testing the model

### 1. Clone the repository:
```
!git clone https://github.com/abbylagar/objectdetection.git
```

### 2. Go to your project directory
```
%cd <project_folder>/objectdetection/KWS
```

### 3. Install the requirements
```
!pip install -r requirements.txt
```


### 4. Train the model
Run below code: 
```
python3 train.py
```
Sample run in the train.ipynb .  
If wandb visualization prompts ,
enter 2 (Use an existing W&B account , project name:  KWS)
enter 3 (No visualization)

Max accuracy after 35 epochs 

![image](https://user-images.githubusercontent.com/67377766/170887714-75ce2ffe-c9a9-4d97-ba66-27926c444c66.png)



### 5. Test model
Run below code: 

```
python3 infer.py 
```
Sample video of keyword spotting GUI in outputs folder filename: sample.mp4

![image](https://user-images.githubusercontent.com/67377766/170887639-5c6de0a0-ea00-469f-9d6d-8b8826a9c58e.png)

