#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:14:05 2020

@author: Yan
"""

from PIL import Image
import torch
import torch.nn as nn
import io
import matplotlib.pyplot as plt

def out_put(string,verbose):
    print(string)
    with open(f"verbose.txt","a") as f:
        f.write(string + "\n")

def read_image(zipfile,name):
    return Image.open(io.BytesIO(zipfile.read(name)))


def weighted_binary_cross_entropy(output, target, weights=None):
    loss = weights[0] * ((1 - target) * torch.log(1 - output)) + weights[1] * (target * torch.log(output))
    
    return torch.neg(torch.mean(loss))
    
def train(epochs,lr,net,file_name,training_generator,test_generator,file,weights = None):
    con = []    

    #Może CUDA?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnt = torch.cuda.device_count()

    if cnt > 1:
        out_put(f"use {torch.cuda.device_count()} GPUS",file_name)
        net = nn.DataParallel(net)
    elif cnt == 1:
        out_put(f"use one gpu",file_name)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = (nn.BCELoss() if weights == None else weighted_binary_cross_entropy)

    #used to record the best accuracy
    best_accuracy = 0
    
    for epoch in range(epochs):
        train_loss = 0 
        pred_label = []
        true_label = []

        #Wywoluje __getitem__ z torch.utils.data.Dataset
        # x - BATCH SIZE - MAX DŁUGOŚĆ WIDEO - RGB - 48 X 48 jpg
        # y - BATCH SIZE - LABEL
        # s - BATCH SIZE - MAX DŁUGOŚĆ W BATCHU MINUS DŁUGOŚĆ WIDEO OBECNEGO

        for x, y,s in training_generator:
            #Obcinanie, nie kazde wideo jest tej samej dlugosci
            index = s.min().item()
            s = s - s.min()
            x = x.type(torch.FloatTensor)[:, index:]
            y = y.type(torch.FloatTensor)

            if torch.cuda.device_count() > 0:
                x  = x.to(device)
                y  = y.to(device)
                s = s.to(device)

            # fig, axs = plt.subplots(3)
            # fig.suptitle('H')
            # exported_data_2 = x.cpu().data.numpy()
            # for i in range(3):
            #     h = exported_data_2[0][-1][i]
            #     axs[i].imshow(h)
            # plt.show()

            print("  Train " + str(x.size()) + " " + str(y.size()) + " " + str(s.size()))

            pred = net(x,s)
            
            pred_y = (pred >= 0.5).float().to(device).data
            pred_label.append(pred_y)
            true_label.append(y)

            if weights == None:
                loss = loss_func(pred, y)
            else:
                loss = loss_func(pred, y,weights = weights)

            for W in net.parameters():
                loss += 0.001 * W.norm(2)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)
        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        out_put('Epoch: ' + str(epoch+1) + ' | Train Accuracy: ' + str(train_accuracy.item()),file_name)
               
        net.eval()
        
        pred_label = []
        true_label = []
        
        for x, y, s in test_generator:
            index = s.min().item()
            s = s - s.min()
            x = x.type(torch.FloatTensor)[:,index:]
            y = y.type(torch.FloatTensor)

            if torch.cuda.device_count() > 0:
                x = x.to(device)
                y = y.to(device)
                s = s.to(device)

            print("  Test batch: " + str(x.size()) + " " + str(y.size()) + " " + str(s.size()))

            res = net(x,s)
            pred_y = (res >= 0.5).float().to(device).data
            pred_label.append(pred_y)
            true_label.append(y)
        
        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)
        
        test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        con.append([epoch,test_accuracy])
        out_put('Epoch: ' + str(epoch+1) + ' | Train Loss: ' + str(train_loss) + ' | Test Accuracy: ' + str(test_accuracy.item()),file_name)

        if test_accuracy > best_accuracy:
            filepath = f"{file_name}/{file}-{epoch+1:}-{loss}-{test_accuracy}.pt"
            torch.save(net.state_dict(), filepath)
            best_accuracy = test_accuracy  
            
        net.train()

    best_v = max(con,key = lambda x:x[1])
    
    perf = f"Best Accuracy for {file} is {best_v[1]} in epoch {best_v[0]}" + "\n"
    
    out_put(perf,file_name)