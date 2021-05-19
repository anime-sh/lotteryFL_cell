import wandb
import client
from typing import List, Dict, Tuple
import torch.nn.utils.prune as prune
import numpy as np
from utils import create_model, copy_model
import random
import os
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from utils import get_prune_params, average_weights_masks, evaluate, fevaluate, super_prune, \
    log_obj, prune_fixed_amount, fed_avg


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
        self.init_model = create_model(args.dataset, args.arch)
        self.model = copy_model(self.init_model, args.dataset, args.arch)
        self.client_accuracies = np.zeros((self.num_clients, args.comm_rounds))
        self.client_data_num = [len(client.train_loader)
                                for client in self.clients]
        self.client_data_num = np.array(self.client_data_num)

    def aggr(
        self,
        models,
        *args,
        **kwargs
    ):
        avg_model = fed_avg(models=models,
                            dataset=self.args.dataset,
                            arch=self.args.arch,
                            data_nums=self.client_data_num)
        source_buffers = dict(self.model.named_buffers())
        for name, buffer in avg_model.named_buffers():
            buffer.data.copy_(source_buffers[name])
        return avg_model

    def update(
        self,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """

        for i in range(self.args.comm_rounds):

            print('-----------------------------', flush=True)
            print(f'| Communication Round: {i+1}  | ', flush=True)
            print('-----------------------------', flush=True)

            if (self.elapsed_comm_rounds %
                self.args.global_prune_freq) == 0 \
                    and self.args.globalPrune == True:

                # Implement global pruning by pruning aggregated model
                # and then copy initial_params to parameters with pruned model's buffer
                self.prune(self.model)
                self.model = copy_model(self.init_model,
                                        self.args.dataset,
                                        self.args.arch,
                                        source_buff=dict(self.model.named_buffers()))

            # upload model,iniitial_model(optional)
            self.upload()

            # select fraction clients for training (uniform sampling)
            clients_idx = np.random.choice(
                self.num_clients, int(self.args.frac * self.num_clients), replace=False)
            clients = self.clients[clients_idx]

            # update client models
            for client in clients:
                client.update()

            # download local models for selected clients
            models, accs = self.download(clients)
            self.client_accuracies[clients_idx, i] = accs
            # aggregate all models (fed-avg)
            self.model = self.aggr(models)

            avg_accuracy = np.sum(accs)/len(clients_idx)
            print('-----------------------------', flush=True)
            print(f'| Average Accuracy: {avg_accuracy}  | ', flush=True)
            print('-----------------------------', flush=True)

            wandb.log({"client_avg_acc": avg_accuracy})
            self.elapsed_comm_rounds += 1

    def download(
        self,
        clients: List[client.Client],
        *args,
        **kwargs
    ):
        # TODO: parallelize downloading models from clients
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"][0] for upload in uploads]
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
        if self.args.report_verbosity:
            log_obj(eval_log_path1, self.model)
            log_obj(eval_log_path2, self.__dict__)

    def upload(
        self,
        *args,
        **kwargs
    ) -> None:
        """
            Upload global model to clients
        """

        # TODO: parallelize upload to clients (broadcasting stratergy)
        init_model = None if self.elapsed_comm_rounds != 0 \
            else copy_model(self.init_model,
                            self.args.dataset,
                            self.args.arch)
        for client in self.clients:
            model_copy = copy_model(self.model,
                                    self.args.dataset,
                                    self.args.arch)
            client.download(model_copy, init_model)
