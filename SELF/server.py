import torch
import torch.nn as nn
import torch.optim as optim
from torch import Module
import numpy as np
from tabulate import tabulate
from utils import create_model
import os
import random


class Server():
    """
        Central Server
    """

    def __init__(
        self,
        configs,
        clients=[],
        comm_rounds=1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.clients = clients
        self.num_clients = len(self.clients)
        self.model = model
        self.configs = configs
        self.model = create_model(configs.dataset, configs.arch)
        self.elapsed_comm_rounds = 0
        self.client_accuracies = np.zeros(
            (self.args.num_clients, self.args.comm_rounds))

    def aggr(
        self,
        *args,
        **kwargs
    ):
        pass

    def update(
        self,
        globalModel,
        *args,
        **kwargs
    ) -> None:
        """
            Interface to Train
        """
        self.elapsed_comm_rounds += 1

    def learn(
        self,
        *args,
        **kwargs
    ):
        self.model.train()
        for i in range(self.comm_rounds):
            print('-----------------------------', flush=True)
            print(f'| Communication Round: {i+1}  | ', flush=True)
            print('-----------------------------', flush=True)
            # broadcast model
            self.upload(self.model)
            #-------------------------------------------------#
            for client in self.clients():
                client.update()
            #-------------------------------------------------#
            models, accs = self.download(self.clients)
            self.model = aggr(models)

    def download(
        self,
        clients,
        *args,
        **kwargs
    ):
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        return models, accs

    def prune(
        self,
        *args,
        **kwargs
    ):
        """
            Prune self.model
        """

        pass

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
    ) -> torch.Module:
        """
            Upload global model to clients
        """
        pass
