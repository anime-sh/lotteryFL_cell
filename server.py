from utils import create_model, copy_model
import random
import os
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from utils import get_prune_params, average_weights_masks, evaluate, fevaluate, super_prune, log_obj
import numpy as np
import torch.nn.utils.prune as prune
from typing import List, Dict, Tuple
import client


class Server():
    """
        Central Server
    """

    def __init__(
        self,
        args,
        test_loader,
        clients=[],
        comm_rounds=1,
    ):
        super().__init__()
        self.clients = np.array(clients, dtype='object')
        self.num_clients = len(self.clients)
        self.args = args
        self.elapsed_comm_rounds = 0
        self.test_loader = test_loader
        self.init_model = create_model(args.dataset, args.arch)
        self.model = copy_model(self.init_model, args.dataset, args.arch)

    def aggr(
        self,
        models,
        *args,
        **kwargs
    ):
        return average_weights_masks(models=models,
                                     dataset=self.args.dataset,
                                     arch=self.args.arch)

    def update(
        self,
        prune,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """

        self.model.train()
        for i in range(self.args.comm_rounds):
            self.elapsed_comm_rounds += 1
            print('-----------------------------', flush=True)
            print(f'| Communication Round: {i+1}  | ', flush=True)
            print('-----------------------------', flush=True)
            if self.elapsed_comm_rounds % 10 == 0 and prune == True:
                self.prune(self.model)
                print("PRUNED GLOBAL MODEL @ SERVER")
            # broadcast model
            self.upload(self.model)
            #-------------------------------------------------#
            clients_idx = np.random.choice(
                self.num_clients, int(self.args.frac * self.num_clients))
            clients = self.clients[clients_idx]
            #-------------------------------------------------#
            for client in clients:
                client.update(self.elapsed_comm_rounds)
            #-------------------------------------------------#
            models, accs = self.download(clients)
            self.model = self.aggr(models)

            eval_score = self.eval(self.model)
            # if kwargs["verbose"] == 1:
            #     print(f"eval_score = {eval_score['Accuracy']}")

    def download(
        self,
        clients: List[client.Client],
        *args,
        **kwargs
    ):
        # TODO: parallelize downloading models from clients
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        return models, accs

    def prune(
        self,
        model,
        *args,
        **kwargs
    ):
        """
            Prune self.model
        """
        super_prune(model=model,
                    init_model=self.init_model,
                    name="weight",
                    threshold=self.args.prune_threshold,
                    verbose=self.args.prune_verbosity)

    def eval(
        self,
        model,
        *args,
        **kwargs
    ):
        """
            Eval self.model
        """
        return fevaluate(model=model,
                        data_loader=self.test_loader,
                        verbose=True)

    def save(
        self,
        *args,
        **kwargs
    ):
        """
            Save model,meta-info,states
        """
        eval_log_path1 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_model.pickle"
        eval_log_path2 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_dict.pickle"
        if self.args.report_verbose:
            log_obj(eval_log_path1, self.model)
            log_obj(eval_log_path2, self.__dict__)

    def upload(
        self,
        model,
        *args,
        **kwargs
    ) -> None:
        """
            Upload global model to clients
        """

        # TODO: parallelize upload to clients (broadcasting stratergy)
        for client in self.clients:
            model_copy = copy_model(model,
                                    self.args.dataset,
                                    self.args.arch
                                    )
            client.download(model_copy,self.init_model)
