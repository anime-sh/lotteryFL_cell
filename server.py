import wandb
from typing import List, Dict, Tuple
import torch.nn.utils.prune as prune
import numpy as np
import random
import os
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from util import get_prune_params, super_prune, fed_avg, l1_prune, create_model, copy_model, get_prune_summary


class Server():
    """
        Central Server
    """

    def __init__(
        self,
        args,
        model,
        clients
    ):
        super().__init__()
        self.clients = clients
        self.num_clients = len(self.clients)
        self.args = args
        self.model = model
        self.init_model = copy_model(model, self.args.device)
        self.prev_model = copy_model(model, self.args.device)

        self.elapsed_comm_rounds = 0
        self.curr_prune_step = 0.00

    def aggr(
        self,
        models,
        clients,
        *args,
        **kwargs
    ):
        print("----------Averaging Models--------")
        weights_per_client = np.array(
            [client.num_data for client in clients], dtype=np.float32)
        weights_per_client /= np.sum(weights_per_client)
        aggr_model = fed_avg(
            models=models,
            weights=weights_per_client,
            device=self.args.device
        )
        pruned_summary, _, _ = get_prune_summary(aggr_model, name='weight')
        print(tabulate(pruned_summary, headers='keys', tablefmt='github'))

        prune_params = get_prune_params(aggr_model)
        for param, name in prune_params:
            zeroed_weights = torch.eq(
                getattr(param, name).data, 0.00).sum().float()
            prune.l1_unstructured(param, name, int(zeroed_weights))

        return aggr_model

    def update(
        self,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """
        self.elapsed_comm_rounds += 1
        self.prev_model = copy_model(self.model, self.args.device)
        print('-----------------------------', flush=True)
        print(
            f'| Communication Round: {self.elapsed_comm_rounds}  | ', flush=True)
        print('-----------------------------', flush=True)
        _, num_pruned, num_total = get_prune_summary(self.model)

        prune_percent = num_pruned / num_total
        # global_model pruned at fixed freq
        # with a fixed pruning step
        if (self.args.server_prune == True and
            (self.elapsed_comm_rounds % self.args.server_prune_freq) == 0) and \
                (prune_percent < self.args.server_prune_threshold):
                
            # prune the model using super_mask
            self.prune()
            # reinitialize model with std.dev of init_model
            self.reinit()

        client_idxs = np.random.choice(
            self.num_clients, int(
                self.args.frac_clients_per_round*self.num_clients),
            replace=False,
        )
        clients = [self.clients[i] for i in client_idxs]

        # upload model to selected clients
        self.upload(clients)

        # call training loop on all clients
        for client in clients:
            client.update()

        # download models from selected clients
        models, accs = self.download(clients)

        avg_accuracy = np.mean(accs, axis=0, dtype=np.float32)
        print('-----------------------------', flush=True)
        print(f'| Average Accuracy: {avg_accuracy}  | ', flush=True)
        print('-----------------------------', flush=True)
        wandb.log({"client_avg_acc": avg_accuracy,
                  "comm_round": self.elapsed_comm_rounds})

        # compute average-model and (prune it by 0.00 )
        aggr_model = self.aggr(models, clients)

        # copy aggregated-model's params to self.model (keep buffer same)
        self.model = aggr_model

        print('Saving global model')
        torch.save(self.model.state_dict(),
                   f"./checkpoints/server_model_{self.elapsed_comm_rounds}.pt")

    def prune(self):
        if self.args.prune_method == 'l1':
            l1_prune(model=self.model,
                     amount=self.args.server_prune_step,
                     name='weight',
                     verbose=self.args.prune_verbose,
                     glob=False)
        elif self.args.prune_method == 'old_super_mask':
            super_prune(model=self.model,
                        init_model=self.init_model,
                        amount=self.args.server_prune_step,
                        name='weight',
                        verbose=self.args.prune_verbose)
        elif self.args.prune_method == 'new_super_mask':
            super_prune(model=self.model,
                        init_model=self.prev_model,
                        amount=self.args.server_prune_step,
                        name='weight',
                        verbose=self.args.prune_verbose)
        elif self.args.prune_method == 'mix_l1_super_mask':
            if self.elapsed_comm_rounds == self.server_prune_freq:
                super_prune(model=self.model,
                            init_model=self.init_model,
                            amount=self.args.server_prune_step,
                            name='weight',
                            verbose=self.args.prune_verbose)
            else:
                l1_prune(model=self.model,
                         amount=self.args.server_prune_step,
                         name='weight',
                         verbose=self.args.prune_verbose,
                         glob=False)

    def reinit(self):
        if self.args.reinit_method == 'none':
            return

        elif self.args.reinit_method == 'std_dev':
            source_params = dict(self.init_model.named_parameters())
            for name, param in self.model.named_parameters():
                std_dev = torch.std(source_params[name].data)
                param.data.copy_(std_dev*torch.sign(source_params[name].data))

        elif self.args.reinit_method == 'init_weights':
            source_params = dict(self.init_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)

    def download(
        self,
        clients,
        *args,
        **kwargs
    ):
        # downloaded models are non pruned (taken care of in fed-avg)
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        return models, accs

    def save(
        self,
        *args,
        **kwargs
    ):
        # """
        #     Save model,meta-info,states
        # """
        # eval_log_path1 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_model.pickle"
        # eval_log_path2 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_dict.pickle"
        # if self.args.report_verbosity:
        #     log_obj(eval_log_path1, self.model)
        pass

    def upload(
        self,
        clients,
        *args,
        **kwargs
    ) -> None:
        """
            Upload global model to clients
        """
        for client in clients:
            # make pruning permanent and then upload the model to clients
            model_copy = copy_model(self.model, self.args.device)
            init_model_copy = copy_model(self.init_model, self.args.device)

            params = get_prune_params(model_copy, name='weight')
            for param, name in params:
                prune.remove(param, name)

            init_params = get_prune_params(init_model_copy)
            for param, name in init_params:
                prune.remove(param, name)
            # call client method
            client.download(model_copy, init_model_copy)
