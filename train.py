
import matplotlib.pyplot as plt
from util import train
import os
import torch
from data import UVANEMODataGenerator
import numpy as np
from model import DeepSmileNet
import matplotlib.pyplot as plt


def train_UVANEMO(epoch,lr,label_path,frame_path,frequency,batch_size,sub,file_name  = "uvanemo_training"):
    for file in os.listdir(label_path): 
        
        current_path = os.path.join(label_path,file)
        if not os.path.isdir(current_path):
            continue

        #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html# do dataset
        #===== TRAIN =====
        train_labels = os.path.join(current_path,"train.json")
        params = {"label_path": train_labels,
                  "frame_path": frame_path,
                  "frequency" : frequency}
        # * - unpack do pozycyjnych argumentów (po kolei) ZMIENILEM
        dg = UVANEMODataGenerator(**params) #dziedziczy po torch.utils.data.Dataset
        training_generator = torch.utils.data.DataLoader(dg,batch_size=batch_size,shuffle=True)
        #===== =====

        # ===== TEST =====
        test_labels = os.path.join(current_path,"test.json")
        params = {"label_path": test_labels,
                  "frame_path": frame_path,
                  "test": True,
                  "frequency": frequency}
        # ** - unpack do nazwanych argumentów (dictionary)
        dg = UVANEMODataGenerator(**params)
        test_generator = torch.utils.data.DataLoader(dg,batch_size=32,shuffle=True)
        
        train(epoch,lr,DeepSmileNet(re = sub),file_name,training_generator,test_generator,file)
        # ===== =====