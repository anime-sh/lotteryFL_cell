from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import copy_model
from typing import Dict
from pytorch_lightning.metrics.functional import accuracy
import numpy as np
import torch
import torchmetrics

metric = torchmetrics.Accuracy()


class CNN(pl.Module):
    def __init__(self, train_loader, test_loader, client_id, comm_rounds, client_epoch, eita_hat, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.trainloader = train_loader
        self.testloader = test_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros((comm_rounds, client_epoch))
        self.losses = np.zeros((comm_rounds, client_epoch))
        self.prune_rates = np.zeros(comm_rounds)
        self.cur_prune_rate = 0.00
        self.eita = eita_hat
        self.train_preds = []
        self.train_labels = []
        self.test_preds = []
        self.test_labels = []

### pl hooks ###
    def train_dataloader(self):
        return self.trainloader

    def test_dataloader(self):
        return self.testloader

    def forward(self, x):
        x = self.model(x)
        return x  # no need to do softmax here, nn cross entropy has a softmax

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_raw = self(x)
        loss = self.criterion(y_raw, y)
        y_pred = nn.Softmax(y_raw, dim=-1)
        self.train_preds.extend(y_pred)
        self.train_targets.extend(y)
        acc = metric(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True,logger=True)
        return loss

    def train_epoch_end(self, train_losses):
        acc = metric.compute()
        metric.reset()

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, test_):
        pass

#### pl hooks end ####
#### custom functions ####
    def download(self, globalModel, global_initModel, *args, **kwargs):
        """
            Download global model from server
        """
        self.globalModel = globalModel
        self.global_initModel = global_initModel

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        return {
            "model": copy_model(self.model,
                                self.args.dataset,
                                self.args.arch),
            "acc": self.eval_score["Accuracy"]
        }
