import numpy as np
import torch
import torchaudio, torchvision
import os
import matplotlib.pyplot as plt 
import librosa
from torchvision.transforms import ToTensor
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from einops import rearrange
from torch import nn
from argparse import ArgumentParser
import tkinter as tk
import tkinter.messagebox
import pyaudio
import wave
import os
from lit_transformer import LitTransformer


sample_rate = 16000

transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                            n_fft=512,
                                            win_length=None,
                                            hop_length=256,
                                            n_mels=40,
                                            power=2.0)

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

    args = parser.parse_args("")
    return args




class InferAudio:

    def __init__(self, chunk=16000, frmat=pyaudio.paInt32, channels=1, rate=16000, py=pyaudio.PyAudio()):

        # Start Tkinter and set Title
        self.main = tkinter.Tk()
        self.collections = []
        self.main.geometry('1000x300')
        self.main.title('Record')
        self.CHUNK = chunk
        self.FORMAT = frmat
        self.CHANNELS = channels
        self.RATE = rate
        self.p = py
        self.frames = []
        self.st = 1
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        # Set Frames
        self.buttons = tkinter.Frame(self.main, padx=150, pady=50)
        self.l1 = tkinter.Label(self.main, text="Click Start",font=("Helvetica", 50),fg='red')
      

        # Pack Frame
        self.buttons.pack(fill=tk.BOTH)

        self.l1.pack()   


        # Start and Stop buttons
        self.strt_rec = tkinter.Button(self.buttons, width=50, padx=10, pady=5, text='Start', command=lambda: self.start_record())
        self.strt_rec.grid(row=0, column=0, padx=150, pady=5)
        self.stop_rec = tkinter.Button(self.buttons, width=50, padx=10, pady=5, text='Stop', command=lambda: self.stop())
        self.stop_rec.grid(row=1, column=0, columnspan=1, padx=150, pady=5)
        
        #self.l1 = Label(self.main, text="hi")

        
        tkinter.mainloop()

    def start_record(self):
        self.st = 1
        self.frames = []
        #datacollect = []
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        maxi = 0
        while self.st == 1:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            #print("* recording")
            #inference   
            samps = np.frombuffer(data,dtype=np.int32)
            samps =samps.astype(np.float32)

            #print(samps.dtype)
            maxi_new=max(abs(samps))
            

            samps=samps/maxi
            #samps =samps.astype(np.float32)
            #print(samps)
            samps=torch.from_numpy(samps)
            #print(samps.shape)
            """        
            if samps.shape[-1] < self.RATE:
                samps= torch.cat([samps, torch.zeros((1, self.RATE - samps.shape[-1]))], dim=-1)
            elif samps.shape[-1] > self.RATE:
                samps = samps[:,:self.RATE]

            """
                    
            #print(samps.size)
            mel = ToTensor()(librosa.power_to_db(transform(samps).squeeze().numpy(), ref=np.max))
           
            #print(mel.size())
            mel = torch.cat([mel, torch.zeros(1,40,1)],dim=-1)
            mel = mel.unsqueeze(0)
            #print(mel.size())
            mel = rearrange(mel, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=1, p2=32)
            scripted_module = torch.jit.load(model_path)
            pred = torch.argmax(scripted_module(mel), dim=1)

            #print(f"Ground Truth: , Prediction: {idx_to_class[pred.item()]}")
   
            prediction = idx_to_class[pred.item()]

            self.l1.config(text=str(prediction))
            #time.sleep(0)
            self.main.update()

        stream.close()

        wf = wave.open('./outputs/voice_record.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
 
    def stop(self):
       # print('stop recording')
        self.st = 0
        



if __name__ == "__main__":

    args = get_args()
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
            'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
            'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}


    model = LitTransformer(num_classes=37, lr=args.lr, epochs=args.max_epochs, 
                        depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                        patch_dim=4, seqlen=32,)

    model = model.load_from_checkpoint(os.path.join('.\checkpoints', "kws_best_acc.ckpt"))
    model.eval()
    script = model.to_torchscript()

    # save for use in production environment
    model_path = os.path.join('.\checkpoints', "kws_best_acc.pt")
    torch.jit.save(script, model_path)

        
    # KWS GUI .
    guiAUD = InferAudio()

