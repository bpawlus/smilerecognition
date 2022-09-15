import torch
import os
import json
import numpy as np
from PIL import Image
import zipfile
from util import read_image
import matplotlib.pyplot as plt

def zipToName(data):
    data_name = dict()
    
    for file in data.namelist():
        if len(file.split("/")) != 2:
            continue
        name,f = file.split("/")
        if name in data_name:
            data_name[name].append(f)
        else:
            data_name[name] = [f]   
            
    return data_name,max([len(i) for i in data_name.values()])

def loadframes(data_name,data,name,frequency,max_len):
    values = []
    files = [f for f in data_name[name] if f.endswith(".jpg") and f.replace(".jpg","").isnumeric()]
    files = sorted(files, key = lambda x : int(x.replace(".jpg","")))
    last = [files[-1]] if len(files) % frequency >= 2/3 * frequency else []
    start = 0
    #co którąś klatkę weź
    files = np.concatenate([np.array(files)[start::frequency],last])
    
    l  = 0
    
    pad_x = np.zeros((max_len,3,48,48))
    for frame in files:
        
        try:
            cur  = read_image(data, os.path.join(name,frame))
        except:
            cur  = read_image(data, name+"/"+frame)
        resized_asarray = np.asarray(cur.resize((48,48)))
        cur = np.swapaxes(resized_asarray, 2, 0)

        values.append(cur)
        l+=1
    
    x = np.array(values)
    l = max_len - l
    pad_x[l:] = x
    
    return pad_x,l

class UVANEMODataGenerator(torch.utils.data.Dataset):
    def __init__(self,frame_path,label_path, frequency = 5,test = False ,scale = 255):
        
        self.data      = zipfile.ZipFile(frame_path)
        self.data_name = dict()
        self.frequency         = frequency
        self.label_path        = label_path
        self.scale             = scale
        self.test              = test
        self.__dataset_information()
        
    def __dataset_information(self):
        self.data_name,self.max_len = zipToName(self.data)
        self.max_len = self.max_len//self.frequency + 2
        self.numbers_of_data    = 0
        self.index_name_dic = dict()        
        
        with open(self.label_path) as f:
            labels = json.load(f)
        for index,(k,v) in enumerate(labels.items()):
            self.index_name_dic[index] = [k,v]
        self.numbers_of_data = index + 1
                     
                  
    def __len__(self):
        return self.numbers_of_data
    
    def __getitem__(self,idx):
        ids    =  self.index_name_dic[idx]
        name,label = ids
        y = np.zeros(1)
        y[0] = label
        pad_x,l = loadframes(self.data_name, self.data, name, self.frequency, self.max_len)
        
        return pad_x/self.scale,y,l