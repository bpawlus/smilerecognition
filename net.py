import os
from dataset import UVANEMODataGenerator
from model import DeepSmileNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class UVANEMO():
    def __init__(self, epochs, lr, label_path, frame_path, frequency, batch_size, output_model_path, verbose_path, out_plot):
        self.epochs = epochs
        self.lr = lr
        self.label_path = label_path
        self.frame_path = frame_path
        self.frequency = frequency
        self.batch_size = batch_size
        self.output_model_path = output_model_path
        self.verbose_path = verbose_path
        self.out_plot = out_plot
        self.training_generator = None
        self.test_generator = None

    def loss_func_weighted(self, output, target, weights):
        #weighted binary cross entropy
        loss = weights[0] * ((1 - target) * torch.log(1 - output)) + weights[1] * (target * torch.log(output))

        return torch.neg(torch.mean(loss))

    def outverbose(self, info):
        print(info)
        with open(f"{self.verbose_path}", "a") as f:
            f.write(info + "\n")

    def prepareData(self, idx):
        current_path = os.path.join(self.label_path,idx)

        if not os.path.isdir(current_path):
            return

        #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html# do dataset

        #===== TRAIN =====
        train_labels = os.path.join(current_path,"train.json")
        params = {"label_path": train_labels,
                "frame_path": self.frame_path,
                "frequency" : self.frequency}

        # ** - unpack do nazwanych argumentów (dictionary)
        dg = UVANEMODataGenerator(**params) #dziedziczy po torch.utils.data.Dataset
        self.training_generator = torch.utils.data.DataLoader(dg,batch_size=self.batch_size,shuffle=True)
        #===== =====

        # ===== VALIDATE =====
        validate_labels = os.path.join(current_path,"validate.json")
        params = {"label_path": validate_labels,
                "frame_path": self.frame_path,
                "frequency": self.frequency,
                "test": True}

        dg = UVANEMODataGenerator(**params)
        self.validate_generator = torch.utils.data.DataLoader(dg,batch_size=8,shuffle=True)
        # ===== =====

        # ===== TEST =====
        test_labels = os.path.join(current_path,"test.json")
        params = {"label_path": test_labels,
                "frame_path": self.frame_path,
                "frequency": self.frequency,
                "test": True}

        dg = UVANEMODataGenerator(**params)
        self.test_generator = torch.utils.data.DataLoader(dg,batch_size=8,shuffle=True)
        # ===== =====

        self.architecture = DeepSmileNet()

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

        self.architecture.to(device)
        optimizer = torch.optim.Adam(self.architecture.parameters(), lr=self.lr)
        loss_func = (nn.BCELoss() if weights == None else self.loss_func_weighted)

        self.last_epochs = []
        self.last_train_accs = []
        self.last_train_losses = []
        self.last_val_accs = []
        self.last_val_losses = []

        # Najlepsze accuracy z walidacyjnego
        best_accuracy = 0

        train_size = len(self.training_generator.dataset)
        validate_size = len(self.validate_generator.dataset)
        test_size = len(self.test_generator.dataset)

        self.validate_penalty = train_size / validate_size

        self.outverbose(f"train size: {train_size}, validate size: {validate_size}, test size: {test_size}")

        for epoch in range(self.epochs):
            train_loss = 0

            pred_label = []
            true_label = []

            # ===== TRAIN EPOCH =====

            # Wywoluje __getitem__ z torch.utils.data.Dataset
            # x - BATCH SIZE - MAX DŁUGOŚĆ WIDEO - RGB - 48 X 48 jpg
            # y - BATCH SIZE - LABEL
            # s - BATCH SIZE - MAX DŁUGOŚĆ W BATCHU MINUS DŁUGOŚĆ WIDEO OBECNEGO

            for x, y, s in self.training_generator:
                # Obcinanie, nie kazde wideo jest tej samej dlugosci
                index = s.min().item()
                s = s - s.min()
                x = x.type(torch.FloatTensor)[:, index:]
                y = y.type(torch.FloatTensor)

                if torch.cuda.device_count() > 0:
                    x = x.to(device)
                    y = y.to(device)
                    s = s.to(device)

                # fig, axs = plt.subplots(3)
                # fig.suptitle('H')
                # exported_data_2 = x.cpu().data.numpy()
                # for i in range(3):
                #     h = exported_data_2[0][-1][i]
                #     axs[i].imshow(h)
                # plt.show()

                pred = self.architecture(x, s)
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

            pred_label = torch.cat(pred_label, 0)
            true_label = torch.cat(true_label, 0)
            train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
            self.outverbose('Epoch: ' + str(epoch) + ' | Train Accuracy: ' + str(train_accuracy.item()) + ' | Train Loss: ' + str(train_loss))

            self.last_epochs.append(epoch)

            self.last_train_accs.append(train_accuracy.item())
            self.last_train_losses.append(train_loss)

            # ===== =====

            self.architecture.eval() #to samo co self.architecture.train(false)

            # ===== VALIDATE EPOCH =====

            validate_loss = 0

            pred_label = []
            true_label = []

            for x, y, s in self.validate_generator:
                index = s.min().item()
                s = s - s.min()
                x = x.type(torch.FloatTensor)[:, index:]
                y = y.type(torch.FloatTensor)

                if torch.cuda.device_count() > 0:
                    x = x.to(device)
                    y = y.to(device)
                    s = s.to(device)

                #print("  Validate batch: " + str(x.size()) + " " + str(y.size()) + " " + str(s.size()))

                pred = self.architecture(x, s)
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

            # ===== UPDATE MODEL =====

            if validate_accuracy > best_accuracy:
                filepath = f"{self.output_model_path}/{idx}-{epoch:}-{loss}-{validate_accuracy}.pt"
                torch.save(self.architecture.state_dict(), filepath)
                best_accuracy = validate_accuracy


            # ===== =====

            self.architecture.train()

        best_v = max(con, key=lambda x: x[1])

        # ===== TEST =====

        # ===== =====

        self.outverbose(f"Best Accuracy for {idx} is {best_v[1]} in epoch {best_v[0]}" + "\n")

    def evaluate_last(self):
        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")

        ax.set_ylabel("Loss", color="red", fontsize=14)
        line1, = ax.plot(self.last_epochs, self.last_train_losses, color="red", marker="o")
        line2, = ax.plot(self.last_epochs, self.last_val_losses, color="orange", marker="o")

        ax2 = ax.twinx()
        ax2.set_ylabel("Acc", color="blue", fontsize=14)
        line3, = ax2.plot(self.last_epochs, self.last_train_accs, color="blue", marker="o")
        line4, = ax2.plot(self.last_epochs, self.last_val_accs, color="cyan", marker="o")

        validate_penalty_str = "Loss Validate (*{value:.2f})"
        ax2.legend([line1, line2, line3, line4], ['Loss Train', validate_penalty_str.format(value=self.validate_penalty), 'Accuracy Train', 'Accuracy Validate'], loc='upper center')

        fig.savefig(self.out_plot, format='jpeg', dpi=100, bbox_inches='tight')
        plt.show()