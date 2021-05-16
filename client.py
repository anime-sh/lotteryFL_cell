import torch
import torch.nn as nn
import torch.optim as optim
from torch import Module
import numpy as np
import os
from utils import copy_model, create_model, get_prune_summary
from typing import Dict, OrderedDict


class Client():
    def __init__(
        self,
        configs,
        *args,
        **kwargs
    ):
        super().__init__()
        self.configs = configs
        self.model = create_model(configs.dataset, configs.arch)

    def update(
        self,
        *args,
        **kwargs
    ) -> None:
        """
            Interface to Server
        """
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params

        self.eval_score = self.eval(self.globalModel)

        if self.eval_score["Accuracy"][0] > self.configs.eita:
            prune_rate = min(cur_prune_rate + self.configs.prune_step,
                             self.configs.prune_percent)
            self.prune(self.globalmodel, prune_rate)
            self.model = copy_model(self.global_initModel,
                                    self.configs.dataset,
                                    self.configs.arch,
                                    dict(self.globalModel.named_buffers()))
            self.configs.eita = self.configs.eita_hat
        else:
            self.configs.eita *= self.configs.alpha
            self.model = copy_model(self.globalModel,
                                    self.configs.dataset,
                                    self.configs.arch)

        #-----------------------TRAINING LOOOP ------------------------#
        train_score = self.train()
        #--------------------------------------------------------------#

        self.save(model)

    def train(
        self,
    ):
        """
            Train NN
        """
        pass

    def prune(
        self,
        model,
        prune_rate,
        *args,
        **kwargs
    ):
        """
            Prune self.model
        """

        pass

    def download(
        self,
        globalModel,
        global_initModel,
        *args,
        **kwargs
    ):
        """
            Download global model from server
        """
        self.globalModel = globalModel
        self.global_initModel = global_initModel

    def eval(
        self,
        *args,
        **kwargs
    ):
        """
            Eval self.model
        """
        pass

    def save(
        self,
        *args,
        **kwargs
    ):
        """
            Save model,meta-info,states
        """
        pass

    def upload(
        self,
        *args,
        **kwargs
    ) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        return {"model": copy_model(self.model,
                                    self.configs.dataset,
                                    self.configs.dataset),
                "acc": self.eval_score["Accuracy"]
                }
