import os
from dataset import UVANEMODataGenerator
from model import DeepSmileNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import json
import shutil
import pandas as pd
import collections

class UVANEMO():
    def __init__(self, epochs, lr, label_path, folds_path, folds_orig_path, videos_path, videos_frequency, features_path, vgg_path,
                 batch_size_train, batch_size_valtest, models_path, calcminmax_features, f):
        self.epochs = epochs
        self.lr = lr
        self.batch_size_train = batch_size_train
        self.batch_size_valtest = batch_size_valtest
        self.videos_frequency = videos_frequency
        self.calcminmax_features = calcminmax_features
        self.f = f

        self.label_path = label_path
        self.folds_path = folds_path
        self.folds_orig_path = folds_orig_path
        self.models_path = models_path

        self.videos_path = videos_path
        self.features_path = features_path
        self.vgg_path = vgg_path

    def loss_func_weighted(self, output, target, weights):
        # weighted binary cross entropy
        loss = weights[0] * ((1 - target) * torch.log(1 - output)) + weights[1] * (target * torch.log(output))

        return torch.neg(torch.mean(loss))

    def outverbose(self, info):
        print(info)
        with open(f"{self.out_verbose}", "a") as f:
            f.write(info + "\n")

    def __determine_value(self, name):
        if "deliberate" in name:
            return 0
        elif "spontaneous" in name:
            return 1
        else:
            return -1

    def split_orig(self, test_fold):
        if not os.path.isdir(self.label_path):
            return

        for i in range(9):
            try:
                shutil.rmtree(os.path.join(self.folds_path, str(i + 1)))
                os.makedirs(os.path.join(self.folds_path, str(i + 1)))
            except FileNotFoundError:
                os.makedirs(os.path.join(self.folds_path, str(i + 1)))

            fold_train_labels = dict()
            fold_validate_labels = dict()
            fold_test_labels = dict()

            train_folds = [i+1 for i in range(10)]
            train_folds.remove(test_fold)
            validate_fold = train_folds[i]
            train_folds.remove(validate_fold)

            for f in train_folds:
                for ps in ["P", "S"]:
                    origfolds = os.path.join(self.folds_orig_path, f"{ps}_fold_all_{f}.txt")
                    with open(origfolds, "r") as readfile:
                        lines = readfile.readlines()
                        for line in lines:
                            line = line.replace("\n", "")
                            fold_train_labels.update({line: self.__determine_value(line)})
            fold_train_labels = collections.OrderedDict(sorted(fold_train_labels.items()))

            for ps in ["P", "S"]:
                origfolds = os.path.join(self.folds_orig_path, f"{ps}_fold_all_{validate_fold}.txt")
                with open(origfolds, "r") as readfile:
                    lines = readfile.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        fold_validate_labels.update({line: self.__determine_value(line)})
                fold_validate_labels = collections.OrderedDict(sorted(fold_validate_labels.items()))

                origfolds = os.path.join(self.folds_orig_path, f"{ps}_fold_all_{test_fold}.txt")
                with open(origfolds, "r") as readfile:
                    lines = readfile.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        fold_test_labels.update({line: self.__determine_value(line)})
                fold_test_labels = collections.OrderedDict(sorted(fold_test_labels.items()))


            fold_train_dir = os.path.join(self.folds_path, str(i + 1), "train.json")
            with open(fold_train_dir, "w") as outfile:
                json.dump(fold_train_labels, outfile, indent=4)

            fold_validate_dir = os.path.join(self.folds_path, str(i + 1), "validate.json")
            with open(fold_validate_dir, "w") as outfile:
                json.dump(fold_validate_labels, outfile, indent=4)

            fold_test_dir = os.path.join(self.folds_path, str(i + 1), "test.json")
            with open(fold_test_dir, "w") as outfile:
                json.dump(fold_test_labels, outfile, indent=4)

            print(f"Fold rest: {len(fold_train_labels)%self.batch_size_train} {len(fold_validate_labels)%self.batch_size_valtest} {len(fold_test_labels)%self.batch_size_valtest}")

    def split(self, k):
        if not os.path.isdir(self.label_path):
            return

        splits = KFold(n_splits=k, shuffle=True, random_state=50)

        train_labels = os.path.join(self.label_path, "train.json")
        with open(train_labels) as f:
            labels_dict = json.load(f)
            labels_list = list(labels_dict)

        fold_splits = splits.split(np.arange(len(labels_list)))
        for fold, split in enumerate(fold_splits):
            fold_train_labels = dict()
            for di in [{labels_list[i]: labels_dict[labels_list[i]]} for i in split[0]]:
                fold_train_labels.update(di)

            fold_validate_labels = dict()
            for di in [{labels_list[i]: labels_dict[labels_list[i]]} for i in split[1]]:
                fold_validate_labels.update(di)

            try:
                shutil.rmtree(os.path.join(self.folds_path, str(fold + 1)))
                os.makedirs(os.path.join(self.folds_path, str(fold + 1)))
            except FileNotFoundError:
                os.makedirs(os.path.join(self.folds_path, str(fold + 1)))

            fold_train_dir = os.path.join(self.folds_path, str(fold + 1), "train.json")
            with open(fold_train_dir, "w") as outfile:
                json.dump(fold_train_labels, outfile, indent=4)

            fold_validate_dir = os.path.join(self.folds_path, str(fold + 1), "validate.json")
            with open(fold_validate_dir, "w") as outfile:
                json.dump(fold_validate_labels, outfile, indent=4)

            fold_test_dir = os.path.join(self.folds_path, str(fold + 1), "test.json")
            shutil.copy2(os.path.join(self.label_path, "test.json"), fold_test_dir)

            # self.outverbose(f"train size: {len(split[0])}, validate size: {len(split[1])}, test size: {0}")

        return

    def prepareData(self, idx):
        current_path = os.path.join(self.folds_path, idx)

        if not os.path.isdir(current_path):
            return

        os.makedirs(os.path.join(self.models_path, str(idx)))
        os.makedirs(os.path.join(self.models_path, str(idx), "models"))

        self.out_evaldata = os.path.join(self.models_path, str(idx), "model")
        self.out_verbose = os.path.join(self.models_path, str(idx), "verbose.txt")
        self.out_model = os.path.join(self.models_path, str(idx), "models")
        self.out_csv_labels = os.path.join(self.models_path, str(idx), "labels.csv")

        # ===== TRAIN =====
        train_labels = os.path.join(current_path, "train.json")
        params = {"label_path": train_labels,
                  "videos_path": self.videos_path,
                  "videos_frequency": self.videos_frequency,
                  "features_path": self.features_path,
                  "vgg_path": self.vgg_path,
                  "calcminmax_features": self.calcminmax_features,
                  "feature_list": self.f
                  }

        # ** - unpack do nazwanych argumentów (dictionary)
        self.train_generator = UVANEMODataGenerator(**params)  # dziedziczy po torch.utils.data.Dataset
        self.train_loader = DataLoader(self.train_generator, batch_size=self.batch_size_train, shuffle=True)
        self.train_size = len(self.train_generator)
        shutil.copy2(os.path.join(current_path, "train.json"), os.path.join(self.models_path, str(idx), "train.json"))

        # ===== VALIDATE =====
        validate_labels = os.path.join(current_path, "validate.json")
        params = {"label_path": validate_labels,
                  "videos_path": self.videos_path,
                  "videos_frequency": self.videos_frequency,
                  "features_path": self.features_path,
                  "vgg_path": self.vgg_path,
                  "feature_list": self.f,
                  "test": True}

        self.validate_generator = UVANEMODataGenerator(**params)
        self.validate_loader = DataLoader(self.validate_generator, batch_size=self.batch_size_valtest, shuffle=True)
        self.validate_size = len(self.validate_generator)
        shutil.copy2(os.path.join(current_path, "validate.json"), os.path.join(self.models_path, str(idx), "validate.json"))
        # ===== =====

        # ===== TEST =====
        test_labels = os.path.join(current_path, "test.json")
        params = {"label_path": test_labels,
                  "videos_path": self.videos_path,
                  "videos_frequency": self.videos_frequency,
                  "features_path": self.features_path,
                  "vgg_path": self.vgg_path,
                  "feature_list": self.f,
                  "test": True}

        self.test_generator = UVANEMODataGenerator(**params)
        self.test_loader = DataLoader(self.test_generator, batch_size=self.batch_size_valtest, shuffle=True)
        self.test_size = len(self.test_generator)
        shutil.copy2(os.path.join(current_path, "test.json"), os.path.join(self.models_path, str(idx), "test.json"))
        # ===== =====

        self.validate_penalty = self.train_size / self.validate_size
        self.test_penalty = self.train_size / self.test_size
        self.architecture = DeepSmileNet(self.f)

    def __train_forward(self, device, data_loader, weights, train, optimizer, loss_func):
        total_loss = 0
        pred_label = []
        true_label = []
        label_idx = []
        names = []

        for idx, name, x, y, s, dynamics_features, frames_len in data_loader:
            # Obcinanie, nie kazde wideo jest tej samej dlugosci

            index = s.min().item()
            s = s - s.min()
            x = x.type(torch.FloatTensor)[:, index:]
            y = y.type(torch.FloatTensor)
            if torch.cuda.device_count() > 0:
                x = x.to(device)
                y = y.to(device)
                s = s.to(device)

            keys = list(dynamics_features.keys())
            for key in keys:
                try:
                    dynamics_features[key].size(dim=2)
                    dynamics_features[key] = dynamics_features[key].type(torch.FloatTensor)[:, :]
                    if torch.cuda.device_count() > 0:
                        dynamics_features[key] = dynamics_features[key].to(device)
                except IndexError:
                    del dynamics_features[key]

            pred = self.architecture(x, s, dynamics_features, frames_len)
            pred_y = (pred >= 0.5).float().to(device).data

            pred_label.append(pred_y)
            true_label.append(y)
            label_idx.append(idx)
            names.append(name)

            if weights == None:
                loss = loss_func(pred, y)
            else:
                loss = loss_func(pred, y, weights=weights)

            for W in self.architecture.parameters():
                loss += 0.001 * W.norm(2)

            total_loss += loss.item()
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return (total_loss, pred_label, true_label, label_idx, names)

    def train(self, idx, weights=None):
        # Może CUDA?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cnt = torch.cuda.device_count()

        if cnt > 1:
            self.outverbose(f"use {torch.cuda.device_count()} GPUS")
            self.architecture = nn.DataParallel(self.architecture)
        elif cnt == 1:
            self.outverbose(f"use one gpu")
        self.outverbose(f"using features: {self.f}")
        self.outverbose(f"using fold: {idx}")
        self.outverbose(f"train size: {self.train_size}, validate size: {self.validate_size}, test size: {self.test_size}")

        self.architecture.to(device)
        optimizer = torch.optim.Adam(self.architecture.parameters(), lr=self.lr)
        loss_func = (nn.BCELoss() if weights == None else self.loss_func_weighted)

        self.last_epochs = []

        self.last_train_accs = []
        self.last_train_losses = []
        self.last_train_pred_labels = {}
        self.last_train_true_labels = {}

        self.last_val_accs = []
        self.last_val_losses = []
        self.last_val_pred_labels = {}
        self.last_val_true_labels = {}

        self.last_test_accs = []
        self.last_test_losses = []
        self.last_test_pred_labels = {}
        self.last_test_true_labels = {}

        # Najlepsze accuracy z walidacyjnego
        best_accuracy = 0

        ds = self.train_loader.dataset
        for i, (k, v) in enumerate(ds.index_name_dic.items()):
            self.last_train_pred_labels[v[0]] = []
            self.last_train_true_labels[v[0]] = [v[1]]

        ds = self.validate_loader.dataset
        for i, (k, v) in enumerate(ds.index_name_dic.items()):
            self.last_val_pred_labels[v[0]] = []
            self.last_val_true_labels[v[0]] = [v[1]]

        ds = self.test_loader.dataset
        for i, (k, v) in enumerate(ds.index_name_dic.items()):
            self.last_test_pred_labels[v[0]] = []
            self.last_test_true_labels[v[0]] = [v[1]]

        for epoch in range(self.epochs):
            self.last_epochs.append(epoch)
            # ===== TRAIN =====

            # Wywoluje __getitem__ z torch.utils.data.Dataset
            # x - BATCH SIZE - MAX DŁUGOŚĆ WIDEO - RGB - 48 X 48 jpg
            # y - BATCH SIZE - LABEL
            # s - BATCH SIZE - MAX DŁUGOŚĆ W BATCHU MINUS DŁUGOŚĆ WIDEO OBECNEGO  (DO CONVLSTM POTRZEBNE TAKIE WARTOSCI

            (train_loss, pred_label, true_label, label_idx, names) = self.__train_forward(device, self.train_loader, weights, True, optimizer, loss_func)

            pred_label = torch.cat(pred_label, 0)
            true_label = torch.cat(true_label, 0)
            names =[item for sublist in names for item in sublist]

            train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
            self.outverbose(
                'Epoch: ' + str(epoch) +
                ' | Train Accuracy: ' + str(train_accuracy.item()) +
                ' | Train Loss: ' + str(train_loss)
            )
            self.last_train_accs.append(train_accuracy.item())
            self.last_train_losses.append(train_loss)
            for i in range(len(names)):
                val = None
                if torch.cuda.device_count() > 0:
                    val = int(pred_label[i].cpu().item())
                else:
                    val = int(pred_label[i].item())
                self.last_train_pred_labels[names[i]].append(val)

            # ===== =====

            self.architecture.eval()  # to samo co self.architecture.train(false)

            # ===== VALIDATE =====

            (validate_loss, pred_label, true_label, label_idx, names) = self.__train_forward(device, self.validate_loader, weights, False, optimizer, loss_func)

            pred_label = torch.cat(pred_label, 0)
            true_label = torch.cat(true_label, 0)
            names = [item for sublist in names for item in sublist]

            validate_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
            self.outverbose(
                'Epoch: ' + str(epoch) +
                ' | Validate Accuracy: ' + str( validate_accuracy.item()) +
                ' | Validate Loss (NORMALIZED): ' + str(validate_loss * self.validate_penalty)
            )
            self.last_val_accs.append(validate_accuracy.item())
            self.last_val_losses.append(validate_loss * self.validate_penalty)
            for i in range(len(names)):
                val = None
                if torch.cuda.device_count() > 0:
                    val = int(pred_label[i].cpu().item())
                else:
                    val = int(pred_label[i].item())
                self.last_val_pred_labels[names[i]].append(val)
            # ===== =====

            # ===== TEST =====

            (test_loss, pred_label, true_label, label_idx, names) = self.__train_forward(device, self.test_loader, weights, False, optimizer, loss_func)

            pred_label = torch.cat(pred_label, 0)
            true_label = torch.cat(true_label, 0)
            names = [item for sublist in names for item in sublist]

            test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
            self.outverbose(
                'Epoch: ' + str(epoch) +
                ' | Test Accuracy: ' + str(test_accuracy.item()) +
                ' | Test Loss (NORMALIZED): ' + str(test_loss * self.test_penalty)
            )
            self.last_test_accs.append(test_accuracy.item())
            self.last_test_losses.append(test_loss * self.test_penalty)
            for i in range(len(names)):
                val = None
                if torch.cuda.device_count() > 0:
                    val = int(pred_label[i].cpu().item())
                else:
                    val = int(pred_label[i].item())
                self.last_test_pred_labels[names[i]].append(val)

            # ===== =====

            # ===== UPDATE MODEL =====
            m_name_value_loss = "{value:.2f}"
            m_name_value_acc = "{value:.4f}"
            m_epoch = f"E-{epoch}"
            m_va = f"VA-{m_name_value_acc.format(value=validate_accuracy)}"
            m_vl = f"VL-{m_name_value_loss.format(value=validate_loss*self.validate_penalty)}"
            m_ta = f"TA-{m_name_value_acc.format(value=test_accuracy)}"
            m_tl = f"TL-{m_name_value_loss.format(value=test_loss*self.test_penalty)}"

            filepath = os.path.join(self.out_model, f"{m_epoch}_{m_va}_{m_vl}_{m_ta}_{m_tl}.pt")
            torch.save(self.architecture.state_dict(), filepath)

            if validate_accuracy > best_accuracy:
                best_accuracy = validate_accuracy

            self.update_training_diagram()
            self.update_csv_data()
            self.architecture.train()
            # ===== =====

    def update_training_diagram(self):
        out_acc = self.out_evaldata + "_acc.jpeg"
        out_loss = self.out_evaldata + "_loss.jpeg"

        if os.path.exists(out_acc):
            os.remove(out_acc)
        if os.path.exists(out_loss):
            os.remove(out_loss)

        validate_penalty_str = "Loss Validate (*{value:.2f})"
        test_penalty_str = "Loss Test (*{value:.2f})"

        # ===== LOSSES =====
        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")

        ax.set_ylabel("Loss", color="red", fontsize=14)
        line1, = ax.plot(self.last_epochs, self.last_train_losses, color="#FA9696", marker="o")
        line2, = ax.plot(self.last_epochs, self.last_val_losses, color="#AF4B4B", marker="o")
        line3, = ax.plot(self.last_epochs, self.last_test_losses, color="#640000", marker="o")

        ax.legend(
            [line1, line2, line3],
            ['Loss Train', validate_penalty_str.format(value=self.validate_penalty),
             test_penalty_str.format(value=self.test_penalty)],
            loc='upper center'
        )
        fig.savefig(out_loss, format='jpeg', dpi=100, bbox_inches='tight')
        #plt.clf()
        plt.close(fig)
        # ===== =====

        # ===== ACCRUACIES =====
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel("Epoch")

        ax2.set_ylabel("Acc", color="blue", fontsize=14)
        line4, = ax2.plot(self.last_epochs, self.last_train_accs, color="#9696FA", marker="o")
        line5, = ax2.plot(self.last_epochs, self.last_val_accs, color="#4B4BAF", marker="o")
        line6, = ax2.plot(self.last_epochs, self.last_test_accs, color="#000064", marker="o")

        ax2.legend(
            [line4, line5, line6],
            ['Accuracy Train', 'Accuracy Validate', 'Accuracy Test'],
            loc='upper center'
        )
        fig2.savefig(out_acc, format='jpeg', dpi=100, bbox_inches='tight')
        #plt.clf()
        plt.close(fig2)
        # ===== =====

    def __update_csv_labels(self, filename, pred_labels, true_labels):
        df = pd.DataFrame.from_dict(pred_labels).transpose()
        df.columns = [f"E{int(name)}" for name in list(df.columns.values)]
        df2 = pd.DataFrame.from_dict(true_labels).transpose()
        df2.columns = ["true"]
        df = df2.join(df)

        df.to_csv(filename, sep=",")

    def update_csv_data(self):
        out_acc = self.out_evaldata + "_acc.csv"
        out_loss = self.out_evaldata + "_loss.csv"

        if os.path.exists(out_acc):
            os.remove(out_acc)
        if os.path.exists(out_loss):
            os.remove(out_loss)

        validate_penalty_str = "Loss Validate (*{value:.2f})"
        test_penalty_str = "Loss Test (*{value:.2f})"

        a = np.asarray([self.last_epochs, self.last_train_losses, self.last_val_losses, self.last_test_losses])
        df = pd.DataFrame(a).transpose()
        df.columns = ['Epoch', 'Loss Train', validate_penalty_str.format(value=self.validate_penalty), test_penalty_str.format(value=self.test_penalty)]
        df.to_csv(out_loss, index=False, sep=",")

        a = np.asarray([self.last_epochs, self.last_train_accs, self.last_val_accs, self.last_test_accs])
        df = pd.DataFrame(a).transpose()
        df.columns = ['Epoch', 'Accuracy Train', 'Accuracy Validate', 'Accuracy Test']
        df.to_csv(out_acc, index=False, sep=",")


        out_labels_train = self.out_evaldata + "_train_labels.csv"
        out_labels_val = self.out_evaldata + "_val_labels.csv"
        out_labels_test = self.out_evaldata + "_test_labels.csv"

        if os.path.exists(out_labels_train):
            os.remove(out_labels_train)
        if os.path.exists(out_labels_val):
            os.remove(out_labels_val)
        if os.path.exists(out_labels_test):
            os.remove(out_labels_test)

        self.__update_csv_labels(out_labels_train, self.last_train_pred_labels, self.last_train_true_labels)
        self.__update_csv_labels(out_labels_val, self.last_val_pred_labels, self.last_val_true_labels)
        self.__update_csv_labels(out_labels_test, self.last_test_pred_labels, self.last_test_true_labels)