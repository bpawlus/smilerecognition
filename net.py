import os

from dataset import UVANEMODataGenerator
from model import DeepSmileNet
from model import MultipleDeepSmileNet
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
    def __init__(self, epochs, lr, folds_path, videos_path, videos_frequency, features_path, vgg_path,
                 batch_size_train, batch_size_valtest, calcminmax_features):
        self.epochs = epochs
        self.lr = lr
        self.batch_size_train = batch_size_train
        self.batch_size_valtest = batch_size_valtest
        self.videos_frequency = videos_frequency
        self.calcminmax_features = calcminmax_features

        self.folds_path = folds_path

        self.videos_path = videos_path
        self.features_path = features_path
        self.vgg_path = vgg_path

    # def loss_func_weighted(self, output, target, weights):
    #     # weighted binary cross entropy
    #     loss = weights[0] * ((1 - target) * torch.log(1 - output)) + weights[1] * (target * torch.log(output))
    #
    #     return torch.neg(torch.mean(loss))

    def out(self, info, file):
        print(info)

        with open(f"{file}", "a") as f:
            f.write(info + "\n")

    def outverbose(self, info):
        self.out(info, self.out_verbose)


    def __determine_value(self, name):
        if "deliberate" in name:
            return 0
        elif "spontaneous" in name:
            return 1
        else:
            return -1

    def split_orig(self, label_path, folds_path):
        for i in range(10):
            try:
                shutil.rmtree(os.path.join(folds_path, str(i + 1)))
                os.makedirs(os.path.join(folds_path, str(i + 1)))
            except FileNotFoundError:
                os.makedirs(os.path.join(folds_path, str(i + 1)))

            fold_train_labels = dict()
            fold_validate_labels = dict()
            fold_test_labels = dict()

            train_folds = [i+1 for i in range(10)]

            validate_fold = train_folds[i%10]
            test_fold = train_folds[(i+1)%10]

            train_folds.remove(test_fold)
            train_folds.remove(validate_fold)

            print(f"  {train_folds} {validate_fold} {test_fold}")

            for f in train_folds:
                for ps in ["P", "S"]:
                    origfolds = os.path.join(label_path, f"{ps}_fold_all_{f}.txt")
                    with open(origfolds, "r") as readfile:
                        lines = readfile.readlines()
                        for line in lines:
                            line = line.replace("\n", "")
                            fold_train_labels.update({line: self.__determine_value(line)})
            fold_train_labels = collections.OrderedDict(sorted(fold_train_labels.items()))

            for ps in ["P", "S"]:
                origfolds = os.path.join(label_path, f"{ps}_fold_all_{validate_fold}.txt")
                with open(origfolds, "r") as readfile:
                    lines = readfile.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        fold_validate_labels.update({line: self.__determine_value(line)})
                fold_validate_labels = collections.OrderedDict(sorted(fold_validate_labels.items()))

                origfolds = os.path.join(label_path, f"{ps}_fold_all_{test_fold}.txt")
                with open(origfolds, "r") as readfile:
                    lines = readfile.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        fold_test_labels.update({line: self.__determine_value(line)})
                fold_test_labels = collections.OrderedDict(sorted(fold_test_labels.items()))


            fold_train_dir = os.path.join(folds_path, str(i + 1), "train.json")
            with open(fold_train_dir, "w") as outfile:
                json.dump(fold_train_labels, outfile, indent=4)

            fold_validate_dir = os.path.join(folds_path, str(i + 1), "validate.json")
            with open(fold_validate_dir, "w") as outfile:
                json.dump(fold_validate_labels, outfile, indent=4)

            fold_test_dir = os.path.join(folds_path, str(i + 1), "test.json")
            with open(fold_test_dir, "w") as outfile:
                json.dump(fold_test_labels, outfile, indent=4)

            print(f"Fold rest: {len(fold_train_labels)%self.batch_size_train} {len(fold_validate_labels)%self.batch_size_valtest} {len(fold_test_labels)%self.batch_size_valtest}")


    def prepare_loaders(self, loader_data_path, features):
        # ===== TRAIN =====
        train_labels = os.path.join(loader_data_path, "train.json")
        params = {"label_path": train_labels,
                  "videos_path": self.videos_path,
                  "videos_frequency": self.videos_frequency,
                  "features_path": self.features_path,
                  "vgg_path": self.vgg_path,
                  "calcminmax_features": self.calcminmax_features,
                  "feature_list": features
                  }

        # ** - unpack do nazwanych argumentów (dictionary)
        self.train_generator = UVANEMODataGenerator(**params)  # dziedziczy po torch.utils.data.Dataset
        self.train_loader = DataLoader(self.train_generator, batch_size=self.batch_size_train, shuffle=True)
        self.train_size = len(self.train_generator)

        # ===== VALIDATE =====
        validate_labels = os.path.join(loader_data_path, "validate.json")
        params = {"label_path": validate_labels,
                  "videos_path": self.videos_path,
                  "videos_frequency": self.videos_frequency,
                  "features_path": self.features_path,
                  "vgg_path": self.vgg_path,
                  "feature_list": features,
                  "test": True}

        self.validate_generator = UVANEMODataGenerator(**params)
        self.validate_loader = DataLoader(self.validate_generator, batch_size=self.batch_size_valtest, shuffle=True)
        self.validate_size = len(self.validate_generator)

        # ===== =====

        # ===== TEST =====
        test_labels = os.path.join(loader_data_path, "test.json")
        params = {"label_path": test_labels,
                  "videos_path": self.videos_path,
                  "videos_frequency": self.videos_frequency,
                  "features_path": self.features_path,
                  "vgg_path": self.vgg_path,
                  "feature_list": features,
                  "test": True}

        self.test_generator = UVANEMODataGenerator(**params)
        self.test_loader = DataLoader(self.test_generator, batch_size=self.batch_size_valtest, shuffle=True)
        self.test_size = len(self.test_generator)

        # ===== =====

        self.validate_penalty = self.train_size / self.validate_size
        self.test_penalty = self.train_size / self.test_size


    def copy_loaders(self, loader_data_path, target_path):
        shutil.copy2(os.path.join(loader_data_path, "train.json"), os.path.join(target_path, "train.json"))
        shutil.copy2(os.path.join(loader_data_path, "validate.json"), os.path.join(target_path, "validate.json"))
        shutil.copy2(os.path.join(loader_data_path, "test.json"), os.path.join(target_path, "test.json"))

    def prepare_output_data(self, target_path):
        self.out_verbose = os.path.join(target_path, "verbose.txt")
        self.out_csv_labels = os.path.join(target_path, "labels.csv")

    def prepare_output_models(self, target_path):
        os.makedirs(os.path.join(target_path, "models"))
        self.out_evaldata = os.path.join(target_path, "model")
        self.out_model = os.path.join(target_path, "models")


    def __init_loaders(self, loader_data_path, folder_path, loader_features):
        self.prepare_loaders(loader_data_path, loader_features)
        self.copy_loaders(loader_data_path, folder_path)

    def __init_outputs(self,folder_path):
        self.prepare_output_models(folder_path)
        self.prepare_output_data(folder_path)


    def prepare_data_training(self, idx, working_dir, features):
        loader_data_path = os.path.join(self.folds_path, idx)
        if not os.path.isdir(loader_data_path):
            return

        self.folder_path = os.path.join(working_dir, str(idx))
        os.makedirs(self.folder_path)

        self.__init_loaders(loader_data_path, self.folder_path, features)
        self.__init_outputs(self.folder_path)

        self.architecture = DeepSmileNet(features)

    def prepare_data_load(self, idx, working_dir, state_dir, variant, ignore):
        loader_data_path = os.path.join(self.folds_path, idx)
        if not os.path.isdir(loader_data_path):
            return

        self.variant = variant

        if not os.path.isdir(loader_data_path):
            return

        self.in_models = os.path.join(state_dir, "models", idx)
        features = set()
        realSmileNets = []
        model_states = [f for f in os.listdir(self.in_models) if os.path.isfile(os.path.join(self.in_models, f))]
        for model_state in model_states:
            model_features = model_state.split(".")[0].split("_")

            conflicts = [ignore_feature for ignore_feature in ignore if ignore_feature in model_features]
            if any(conflicts):
                continue
            realSmileNet = DeepSmileNet(model_features)
            checkpoint = torch.load(os.path.join(self.in_models, model_state))
            realSmileNet.load_state_dict(checkpoint, strict=False)
            realSmileNets.append(realSmileNet)

            for model_feature in model_features:
                features.add(model_feature)

        self.folder_path = os.path.join(working_dir, str(idx))
        os.makedirs(self.folder_path)

        self.__init_loaders(loader_data_path, self.folder_path, features)
        self.__init_outputs(self.folder_path)

        self.architecture = MultipleDeepSmileNet(realSmileNets, self.variant)


    def prepare_training_statistics(self):
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

    def save_model(self, e, va, vl, ta, tl):
        m_name_value_loss = "{value:.2f}"
        m_name_value_acc = "{value:.4f}"
        m_epoch = f"E-{e}"
        m_va = f"VA-{m_name_value_acc.format(value=va)}"
        m_vl = f"VL-{m_name_value_loss.format(value=vl)}"
        m_ta = f"TA-{m_name_value_acc.format(value=ta)}"
        m_tl = f"TL-{m_name_value_loss.format(value=tl)}"

        filepath = os.path.join(self.out_model, f"{m_epoch}_{m_va}_{m_vl}_{m_ta}_{m_tl}.pt")
        torch.save(self.architecture.state_dict(), filepath)

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

    def update_csv_lossacc_data(self):
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

    def update_csv_labels_data(self):
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

    def __training_prepare(self, idx):
        # Może CUDA?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cnt = torch.cuda.device_count()

        if cnt > 1:
            self.outverbose(f"use {torch.cuda.device_count()} GPUS")
            self.architecture = nn.DataParallel(self.architecture)
        elif cnt == 1:
            self.outverbose(f"use one gpu")
        self.outverbose(f"using features: {self.train_generator.feature_list}")
        self.outverbose(f"using lr: {self.lr}")
        if idx is not None:
            self.outverbose(f"using fold: {idx}")
        self.outverbose(f"train size: {self.train_size}, validate size: {self.validate_size}, test size: {self.test_size}")

        self.architecture.to(device)
        optimizer = torch.optim.Adam(self.architecture.parameters(), lr=self.lr)
        loss_func = (nn.BCELoss())

        self.prepare_training_statistics()
        return device,optimizer,loss_func

    def __debug_params(self, optimizer, prefix):
        file = os.path.join(self.folder_path, f"{prefix}.txt")

        self.out("Optimizer state_dict:", file)
        for var_name in optimizer.state_dict():
            self.out(f"{var_name}: {optimizer.state_dict()[var_name]}", file)
        self.out("\n", file)

        self.out("Model params:", file)
        for name, param in self.architecture.named_parameters():
            self.out(f"\n{name}"
                            f"\n{param.data}"
                            f"\nTrainable: {param.requires_grad}"
                            , file)

        self.out("Model state_dict:", file)
        for param_tensor in self.architecture.state_dict():
            self.out(f"\n{param_tensor}"
                            f"\n({self.architecture.state_dict()[param_tensor].size()})"
                            f"\n{self.architecture.state_dict()[param_tensor]}"
                            , file)

        self.out("\n", file)


    def __process_loader_data(self, x_video, y, s, x_df_dict, device):
        # Obcinanie, nie kazde wideo jest tej samej dlugosci
        index = s.min().item()
        s = s - s.min()
        x_video = x_video.type(torch.FloatTensor)[:, index:]
        y = y.type(torch.FloatTensor)
        if torch.cuda.device_count() > 0:
            x_video = x_video.to(device)
            y = y.to(device)
            s = s.to(device)

        keys = list(x_df_dict.keys())
        for key in keys:
            try:
                x_df_dict[key].size(dim=2)
                x_df_dict[key] = x_df_dict[key].type(torch.FloatTensor)[:, :]
                if torch.cuda.device_count() > 0:
                    x_df_dict[key] = x_df_dict[key].to(device)
            except IndexError:
                del x_df_dict[key]

        return x_video, y, s, x_df_dict

    def __fit(self, x, s, dynamics_features, frames_len, device):
        pred = self.architecture(x, s, dynamics_features, frames_len)
        pred_y = (pred >= 0.5).float().to(device).data
        return pred, pred_y

    def __calc_loss(self, train, loss_func, pred, y, optimizer):
        loss = loss_func(pred, y)

        for W in self.architecture.parameters():
            loss += 0.001 * W.norm(2)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item()


    def __calc_and_out_total_accloss(self, prefix, epoch, pred_label, true_label, loss, accs_arr, losses_arr):
        accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)

        self.outverbose(
            f'Epoch: ' + str(epoch) +
            f' | {prefix} Accuracy: ' + str(accuracy.item()) +
            f' | {prefix} Loss: ' + str(loss)
        )
        accs_arr.append(accuracy.item())
        losses_arr.append(loss)

        return accuracy, loss

    def __calc_total_labels(self, names, pred_label, labels_arr):
        names = [item for sublist in names for item in sublist]

        for i in range(len(names)):
            val = None
            if torch.cuda.device_count() > 0:
                val = int(pred_label[i].cpu().item())
            else:
                val = int(pred_label[i].item())
            labels_arr[names[i]].append(val)

    def __train_forward(self, device, data_loader, train, optimizer, loss_func):
        total_loss = 0

        pred_label = []
        true_label = []
        label_idx = []
        names = []

        for idx, name, x_video, y, s, x_df_dict, frames_len in data_loader:
            x_video, y, s, x_df_dict = self.__process_loader_data(x_video, y, s, x_df_dict,device)
            pred, pred_y = self.__fit(x_video, s, x_df_dict, frames_len, device)

            pred_label.append(pred_y)
            true_label.append(y)
            label_idx.append(idx)
            names.append(name)

            loss = self.__calc_loss(train, loss_func, pred, y, optimizer)
            total_loss += loss

        pred_label = torch.cat(pred_label, 0)
        true_label = torch.cat(true_label, 0)

        return (total_loss, pred_label, true_label, label_idx, names)

    def __train_evaluate_forward(self, epoch, prefix, accs_arr, losses_arr, labels_arr, data_loader, train, device, optimizer, loss_func, loss_penalty):
        (loss, pred_label, true_label, label_idx, names) = self.__train_forward(device, data_loader, train, optimizer, loss_func)
        loss=loss*loss_penalty

        accuracy, loss = self.__calc_and_out_total_accloss(prefix, epoch, pred_label, true_label, loss, accs_arr, losses_arr)
        self.__calc_total_labels(names, pred_label, labels_arr)

        return accuracy, loss

    def train(self, idx):
        device,optimizer,loss_func = self.__training_prepare(idx)

        for epoch in range(self.epochs):
            self.last_epochs.append(epoch)
            # ===== TRAIN =====

            train_accuracy, train_loss = self.__train_evaluate_forward(epoch, "Train",
                                          self.last_train_accs,
                                          self.last_train_losses,
                                          self.last_train_pred_labels,
                                          self.train_loader,
                                          True, device, optimizer, loss_func, 1)

            # ===== =====

            self.architecture.eval()  # to samo co self.architecture.train(false)

            # ===== VALIDATE =====

            validate_accuracy, validate_loss = self.__train_evaluate_forward(epoch, "Validate",
                                          self.last_val_accs,
                                          self.last_val_losses,
                                          self.last_val_pred_labels,
                                          self.validate_loader,
                                          False, device, optimizer, loss_func, self.validate_penalty)

            # ===== =====

            # ===== TEST =====

            test_accuracy, test_loss = self.__train_evaluate_forward(epoch, "Test",
                                          self.last_test_accs,
                                          self.last_test_losses,
                                          self.last_test_pred_labels,
                                          self.test_loader,
                                          False, device, optimizer, loss_func, self.test_penalty)

            # ===== =====

            # ===== UPDATE MODEL =====
            self.save_model(epoch, validate_accuracy, validate_loss*self.validate_penalty, test_accuracy, test_loss*self.test_penalty)
            self.update_training_diagram()
            self.update_csv_lossacc_data()
            self.update_csv_labels_data()

            self.architecture.train()
            # ===== =====


    def __load_forward(self, device, data_loader, train, optimizer, loss_func):
        total_loss = 0

        pred_label = []
        true_label = []
        label_idx = []
        names = []

        for idx, name, x_videos, y, s, x_df_dict, frames_len in data_loader:
            x_videos, y, s, dynamics_features = self.__process_loader_data(x_videos, y, s, x_df_dict, device)
            pred, predy = self.__fit(x_videos, s, x_df_dict, frames_len, device)

            pred_label.append(predy)
            true_label.append(y)
            label_idx.append(idx)
            names.append(name)

            loss = self.__calc_loss(train, loss_func, pred, y, optimizer)
            total_loss += loss

        pred_label = torch.cat(pred_label, 0)
        true_label = torch.cat(true_label, 0)

        return (total_loss, pred_label, true_label, label_idx, names)

    def __load_evaluate_forward(self, epoch, prefix, accs_arr, losses_arr, labels_arr, data_loader, train, device, optimizer, loss_func, loss_penalty):
        (loss, pred_label, true_label, label_idx, names) = self.__load_forward(device, data_loader, train, optimizer, loss_func)
        loss = loss * loss_penalty

        accuracy, loss = self.__calc_and_out_total_accloss(prefix, epoch, pred_label, true_label, loss, accs_arr, losses_arr)
        self.__calc_total_labels(names, pred_label, labels_arr)

        return accuracy, loss

    def load(self, idx):
        device,optimizer,loss_func = self.__training_prepare(idx)
        self.__debug_params(optimizer, "verbose_model_s")

        self.architecture.train()
        for epoch in range(self.epochs):
            self.last_epochs.append(epoch)

            # ===== TRAIN =====

            train_accuracy, train_loss = self.__load_evaluate_forward(epoch, "Train",
                                          self.last_train_accs,
                                          self.last_train_losses,
                                          self.last_train_pred_labels,
                                          self.train_loader,
                                          True, device, optimizer, loss_func, 1)

            # ===== =====

            self.architecture.eval()  # to samo co self.architecture.train(false)

            # ===== VALIDATE =====

            validate_accuracy, validate_loss = self.__load_evaluate_forward(epoch, "Validate",
                                                                             self.last_val_accs,
                                                                             self.last_val_losses,
                                                                             self.last_val_pred_labels,
                                                                             self.validate_loader,
                                                                             False, device, optimizer, loss_func, self.validate_penalty)

            # ===== =====

            # ===== TEST =====

            test_accuracy, test_loss = self.__load_evaluate_forward(epoch, "Test",
                                                                     self.last_test_accs,
                                                                     self.last_test_losses,
                                                                     self.last_test_pred_labels,
                                                                     self.test_loader,
                                                                     False, device, optimizer, loss_func, self.test_penalty)

            # ===== =====

            # ===== UPDATE MODEL =====
            self.save_model(epoch, validate_accuracy, validate_loss*self.validate_penalty, test_accuracy, test_loss*self.test_penalty)
            self.update_training_diagram()
            self.update_csv_lossacc_data()
            self.update_csv_labels_data()

            self.architecture.train()
            if epoch%10 == 0:
                self.__debug_params(optimizer, f"verbose_model_{str(epoch)}")
            # ===== =====