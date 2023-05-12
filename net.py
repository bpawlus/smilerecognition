import os
from dataset import UVANEMODataGenerator
from model import DeepSmileNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QWidget
from torch.utils.data import DataLoader, SubsetRandomSampler
import sys
from sklearn.model_selection import KFold
import numpy as np

class UVANEMO():
    def __init__(self, epochs, lr, label_path, videos_path, videos_frequency, features_path, vgg_path, batch_size, out_plot, out_verbose, out_model, calcminmax_features, f):
        self.epochs = epochs
        self.lr = lr
        self.label_path = label_path
        self.videos_path = videos_path
        self.videos_frequency = videos_frequency
        self.features_path = features_path
        self.vgg_path = vgg_path
        self.batch_size = batch_size
        self.out_plot = out_plot
        self.out_verbose = out_verbose
        self.out_model = out_model
        self.training_generator = None
        self.test_generator = None
        self.calcminmax_features = calcminmax_features
        self.f = f

        self.__init_app()

    def __init_app(self):
        self.app = QApplication(sys.argv)
        self.window = QWidget()

    def loss_func_weighted(self, output, target, weights):
        #weighted binary cross entropy
        loss = weights[0] * ((1 - target) * torch.log(1 - output)) + weights[1] * (target * torch.log(output))

        return torch.neg(torch.mean(loss))

    def outverbose(self, info):
        print(info)
        with open(f"{self.out_verbose}", "a") as f:
            f.write(info + "\n")

    def prepareData(self, idx):
        current_path = os.path.join(self.label_path,idx)

        if not os.path.isdir(current_path):
            return

        #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html# do dataset

        #===== TRAIN + VALIDATE (TO KFOLD) =====
        train_labels = os.path.join(current_path,"train.json")
        params = {"label_path": train_labels,
                "videos_path": self.videos_path,
                "videos_frequency" : self.videos_frequency,
                "features_path": self.features_path,
                "vgg_path": self.vgg_path,
                "calcminmax_features": self.calcminmax_features,
                "feature_list": self.f
                }

        # ** - unpack do nazwanych argumentów (dictionary)
        self.t_v_generator = UVANEMODataGenerator(**params) #dziedziczy po torch.utils.data.Dataset
        #===== =====


        # ===== TEST =====
        test_labels = os.path.join(current_path,"test.json")
        params = {"label_path": test_labels,
                "videos_path": self.videos_path,
                "videos_frequency": self.videos_frequency,
                "features_path": self.features_path,
                "vgg_path": self.vgg_path,
                "feature_list": self.f,
                "test": True}

        self.test_generator = UVANEMODataGenerator(**params)
        self.test_loader = DataLoader(self.test_generator,batch_size=8,shuffle=True)
        # ===== =====

        self.architecture = DeepSmileNet(self.f)

    def train(self, idx, weights=None):
        con = []
        # Może CUDA?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cnt = torch.cuda.device_count()

        if cnt > 1:
            self.outverbose(f"use {torch.cuda.device_count()} GPUS")
            self.architecture = nn.DataParallel(self.architecture)
        elif cnt == 1:
            self.outverbose(f"use one gpu")
        self.outverbose(f"using features: {self.f}")

        self.architecture.to(device)
        optimizer = torch.optim.Adam(self.architecture.parameters(), lr=self.lr)
        loss_func = (nn.BCELoss() if weights == None else self.loss_func_weighted)

        self.last_fold_end = []
        self.last_epochs = []
        self.last_train_accs = []
        self.last_train_losses = []
        self.last_val_accs = []
        self.last_val_losses = []
        self.last_test_accs = []
        self.last_test_losses = []

        # Najlepsze accuracy z walidacyjnego
        best_accuracy = 0

        train_size = len(self.t_v_generator)
        test_size = len(self.test_loader.dataset)

        k = 10
        splits = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_splits = splits.split(np.arange(train_size))

        for fold, f in enumerate(fold_splits):
            self.outverbose(f"train size: {len(f[0])}, validate size: {len(f[1])}, test size: {test_size}")
            self.validate_penalty = len(f[0]) / len(f[1])
            self.test_penalty = len(f[0]) / test_size
            break



        for fold, (train_idx,val_idx) in enumerate(fold_splits):
            self.outverbose(f"Fold {fold}")

            self.last_fold_end.append(fold*self.epochs)
            self.update_training_diagram()

            training_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            training_loader = DataLoader(self.t_v_generator, batch_size=self.batch_size, sampler=training_sampler)
            validate_loader = DataLoader(self.t_v_generator, batch_size=self.batch_size, sampler=test_sampler)
            #len(training_loader.dataset) zwróci błędny wynik z racji na sampler

            for epoch in range(self.epochs):
                train_loss = 0

                pred_label = []
                true_label = []

                # ===== TRAIN =====

                # Wywoluje __getitem__ z torch.utils.data.Dataset
                # x - BATCH SIZE - MAX DŁUGOŚĆ WIDEO - RGB - 48 X 48 jpg
                # y - BATCH SIZE - LABEL
                # s - BATCH SIZE - MAX DŁUGOŚĆ W BATCHU MINUS DŁUGOŚĆ WIDEO OBECNEGO  (DO CONVLSTM POTRZEBNE TAKIE WARTOSCI


                for x, y, s, aus, si, d1da27, d2da27, frames_len in training_loader:
                    # Obcinanie, nie kazde wideo jest tej samej dlugosci

                    index = s.min().item()
                    s = s - s.min()
                    x = x.type(torch.FloatTensor)[:, index:]
                    y = y.type(torch.FloatTensor)
                    aus = aus.type(torch.FloatTensor)[:, :]
                    si = si.type(torch.FloatTensor)[:, :]
                    d1da27 = d1da27.type(torch.FloatTensor)[:, :]
                    d2da27 = d2da27.type(torch.FloatTensor)[:, :]

                    if torch.cuda.device_count() > 0:
                        x = x.to(device)
                        y = y.to(device)
                        s = s.to(device)
                        aus = aus.to(device)
                        si = si.to(device)
                        d1da27 = d1da27.to(device)
                        d2da27 = d2da27.to(device)

                    pred = self.architecture(x, s, aus, si, d1da27, d2da27, frames_len)
                    pred_y = (pred >= 0.5).float().to(device).data

                    pred_label.append(pred_y)
                    true_label.append(y)

                    if weights == None:
                        loss = loss_func(pred, y)
                    else:
                        loss = loss_func(pred, y, weights=weights)

                    for W in self.architecture.parameters():
                        loss += 0.001 * W.norm(2)

                    train_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    continue

                pred_label = torch.cat(pred_label, 0)
                true_label = torch.cat(true_label, 0)
                train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
                self.outverbose('Epoch: ' + str(epoch) + ' | Train Accuracy: ' + str(train_accuracy.item()) + ' | Train Loss: ' + str(train_loss))

                self.last_epochs.append(fold*self.epochs+epoch)
                self.last_train_accs.append(train_accuracy.item())
                self.last_train_losses.append(train_loss)

                # ===== =====

                self.architecture.eval() #to samo co self.architecture.train(false)

                # ===== VALIDATE =====

                validate_loss = 0

                pred_label = []
                true_label = []

                for x, y, s, aus, si, d1da27, d2da27, frames_len in validate_loader:
                    index = s.min().item()
                    s = s - s.min()
                    x = x.type(torch.FloatTensor)[:, index:]
                    y = y.type(torch.FloatTensor)
                    aus = aus.type(torch.FloatTensor)[:, :]
                    si = si.type(torch.FloatTensor)[:, :]
                    d1da27 = d1da27.type(torch.FloatTensor)[:, :]
                    d2da27 = d2da27.type(torch.FloatTensor)[:, :]

                    if torch.cuda.device_count() > 0:
                        x = x.to(device)
                        y = y.to(device)
                        s = s.to(device)
                        aus = aus.to(device)
                        si = si.to(device)
                        d1da27 = d1da27.to(device)
                        d2da27 = d2da27.to(device)

                    pred = self.architecture(x, s, aus, si, d1da27, d2da27, frames_len)
                    pred_y = (pred >= 0.5).float().to(device).data

                    pred_label.append(pred_y)
                    true_label.append(y)

                    if weights == None:
                        loss = loss_func(pred, y)
                    else:
                        loss = loss_func(pred, y, weights=weights)

                    for W in self.architecture.parameters():
                        loss += 0.001 * W.norm(2)

                    validate_loss += loss.item()

                pred_label = torch.cat(pred_label, 0)
                true_label = torch.cat(true_label, 0)

                validate_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
                con.append([epoch, validate_accuracy])
                self.outverbose('Epoch: ' + str(epoch) + ' | Validate Accuracy: ' + str(validate_accuracy.item()) + ' | Validate Loss (NORMALIZED): ' + str(validate_loss*self.validate_penalty))

                self.last_val_accs.append(validate_accuracy.item())
                self.last_val_losses.append(validate_loss*self.validate_penalty)
            
                # ===== =====

                # ===== TEST =====

                test_loss = 0

                pred_label = []
                true_label = []

                for x, y, s, aus, si, d1da27, d2da27, frames_len in self.test_loader:
                    index = s.min().item()
                    s = s - s.min()
                    x = x.type(torch.FloatTensor)[:, index:]
                    y = y.type(torch.FloatTensor)
                    aus = aus.type(torch.FloatTensor)[:, :]
                    si = si.type(torch.FloatTensor)[:, :]
                    d1da27 = d1da27.type(torch.FloatTensor)[:, :]
                    d2da27 = d2da27.type(torch.FloatTensor)[:, :]

                    if torch.cuda.device_count() > 0:
                        x = x.to(device)
                        y = y.to(device)
                        s = s.to(device)
                        aus = aus.to(device)
                        si = si.to(device)
                        d1da27 = d1da27.to(device)
                        d2da27 = d2da27.to(device)

                    pred = self.architecture(x, s, aus, si, d1da27, d2da27, frames_len)
                    pred_y = (pred >= 0.5).float().to(device).data

                    pred_label.append(pred_y)
                    true_label.append(y)

                    if weights == None:
                        loss = loss_func(pred, y)
                    else:
                        loss = loss_func(pred, y, weights=weights)

                    for W in self.architecture.parameters():
                        loss += 0.001 * W.norm(2)

                    test_loss += loss.item()

                pred_label = torch.cat(pred_label, 0)
                true_label = torch.cat(true_label, 0)

                test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
                con.append([epoch, test_accuracy])
                self.outverbose('Epoch: ' + str(epoch) + ' | Test Accuracy: ' + str(test_accuracy.item()) + ' | Test Loss (NORMALIZED): ' + str(test_loss*self.test_penalty))

                self.last_test_accs.append(test_accuracy.item())
                self.last_test_losses.append(test_loss*self.test_penalty)
                # ===== =====

                # ===== UPDATE MODEL =====

                filepath = f"{self.out_model}/{idx}-F_{fold}-E_{epoch}.pt"
                torch.save(self.architecture.state_dict(), filepath)

                if validate_accuracy > best_accuracy:
                    best_accuracy = validate_accuracy

                self.update_training_diagram()
                self.architecture.train()

                # ===== =====

    def update_training_diagram(self):
        if os.path.exists(self.out_plot):
            os.remove(self.out_plot)
        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")

        ax.set_ylabel("Loss", color="red", fontsize=14)
        line1, = ax.plot(self.last_epochs, self.last_train_losses, color="#FA9696", marker="o")
        line2, = ax.plot(self.last_epochs, self.last_val_losses, color="#AF4B4B", marker="o")
        line3, = ax.plot(self.last_epochs, self.last_test_losses, color="#640000", marker="o")

        ax2 = ax.twinx()
        ax2.set_ylabel("Acc", color="blue", fontsize=14)
        line4, = ax2.plot(self.last_epochs, self.last_train_accs, color="#9696FA", marker="o")
        line5, = ax2.plot(self.last_epochs, self.last_val_accs, color="#4B4BAF", marker="o")
        line6, = ax2.plot(self.last_epochs, self.last_test_accs, color="#000064", marker="o")


        for i in self.last_fold_end:
            plt.axvline(x=i, color='green')

        validate_penalty_str = "Loss Validate (*{value:.2f})"
        test_penalty_str = "Loss Test (*{value:.2f})"
        ax2.legend([line1, line2, line3, line4, line5, line6], ['Loss Train', validate_penalty_str.format(value=self.validate_penalty), test_penalty_str.format(value=self.test_penalty), 'Accuracy Train', 'Accuracy Validate', 'Accuracy Test'], loc='upper center')

        fig.savefig(self.out_plot, format='jpeg', dpi=100, bbox_inches='tight')
        #plt.show()