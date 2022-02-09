import os
from time import sleep

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.deeplog import DeepLog
from utils.earlystop import EarlyStop
from utils.preprocess import preprocess
from utils.tools import str2list


class EXP:
    def __init__(self, g, epochs, batch_size, w, lr, patience, generate, device):
        self.g = g
        self.w = w
        self.lr = lr
        self.generate = generate
        self.batch_size = batch_size
        self.device = device
        self.patience = patience
        self.epochs = epochs

        self.model_path = './checkpoint/checkpoint.pkl'
        self.img_path = './img/loss.jpg'
        self.pred_path = './result/pred.csv'
        self.result_path = './result/result.csv'

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./img/'):
            os.makedirs('./img/')
        if not os.path.exists('./result/'):
            os.makedirs('./result/')

        self._get_data()
        self._get_model()

    def _get_data(self):
        train, valid, test = preprocess(self.generate)
        trainset = MyDataset(train, self.w)
        validset = MyDataset(valid, self.w)
        testset = MyDataset(test, self.w)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        self.loss = {'Train': [], 'Valid': []}

        print('Block || Train: {0} | Valid: {1} | Test: {2}'.format(len(train), len(valid), len(test)))
        print('Dataset || Train: {0} | Valid: {1} | Test: {2}'.format(len(trainset), len(validset), len(testset)))

    def _get_model(self):
        self.model = DeepLog().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.earlystopping = EarlyStop(patience=self.patience)

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.long().to(self.device)

        out = self.model(batch_x)
        loss = self.criterion(out, batch_y)

        return out, loss

    def fit(self):
        # init loss
        self.model.eval()
        train_loss, valid_loss = [], []
        for (batch_X, batch_y, _, _) in tqdm(self.trainloader):
            _, loss = self._process_one_batch(batch_X, batch_y)
            train_loss.append(loss.item())

        for (batch_X, batch_y, _, _) in tqdm(self.validloader):
            _, loss = self._process_one_batch(batch_X, batch_y)
            valid_loss.append(loss.item())

        train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
        print("Init | Train Loss: {0:.6f} Valid Loss: {1:.6f}".format(train_loss, valid_loss))
        sleep(1)

        # train
        for e in range(self.epochs):
            self.model.train()
            train_loss, valid_loss = [], []
            for (batch_X, batch_y, _, _) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                _, loss = self._process_one_batch(batch_X, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            for (batch_X, batch_y, _, _) in tqdm(self.validloader):
                _, loss = self._process_one_batch(batch_X, batch_y)
                valid_loss.append(loss.item())

            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)

            self.loss['Train'].append(train_loss)
            self.loss['Valid'].append(valid_loss)

            print("Epoch {0} | Train Loss: {1:.6f} Valid Loss: {2:.6f}".format(e + 1, train_loss, valid_loss))
            sleep(1)
            self.earlystopping(valid_loss, self.model, self.model_path)
            if self.earlystopping.early_stop:
                print("Early stopping!")
                sleep(1)
                break
        self.model.load_state_dict(torch.load(self.model_path))

        plt.figure()
        plt.plot(self.loss['Train'], label="Train Loss")
        plt.plot(self.loss['Valid'], label="valid Loss")
        plt.title("Losses during training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.img_path, bbox_inches="tight")

    def predict(self, model_load=False, pred_load=False):
        if pred_load:
            result = pd.read_csv(self.pred_path)
            result['Pred'] = result['Pred'].apply(str2list)
        else:
            if model_load:
                self.model.load_state_dict(torch.load(self.model_path))

            self.model.eval()
            idx, label, true, pred = [], [], [], []
            result = pd.DataFrame()
            for (batch_X, batch_y, batch_idx, batch_label) in tqdm(self.testloader):
                out, _ = self._process_one_batch(batch_X, batch_y)
                idx.extend(list(batch_idx))
                label.extend(batch_label.detach().cpu().numpy())
                true.extend(batch_y.detach().cpu().numpy())
                pred.extend(out.detach().cpu().numpy())

            result['BlockId'] = idx
            result['Label'] = label
            result['True'] = true
            result['Pred'] = pred
            result[['BlockId', 'Label', 'True', 'Pred']].to_csv(self.pred_path, index=False)

        result['Pred'] = result.apply(lambda x: 1 - int(x['True'] in x['Pred'].argsort()[-self.g:][::-1]), axis=1)
        result = result[['BlockId', 'Label', 'Pred']].groupby('BlockId').sum()
        result = result.reset_index()
        result['Pred'] = result['Pred'].apply(lambda x: int(x > 0))
        result['Label'] = result['Label'].apply(lambda x: int(x > 0))
        result[['BlockId', 'Label', 'Pred']].to_csv(self.result_path, index=False)

        print("Test || precision: {0:.6f} recall: {1:.6f} f1: {2:.6f}".format(
            precision_score(result['Label'].values, result['Pred'].values),
            recall_score(result['Label'].values, result['Pred'].values),
            f1_score(result['Label'].values, result['Pred'].values)))
