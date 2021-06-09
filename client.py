import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from typing import Dict
import copy
import math
import wandb
from torch.nn.utils import prune
from util import get_prune_summary, l1_prune, get_prune_params, copy_model
from util import train as util_train
from util import test as util_test


class Client():
    def __init__(
        self,
        idx,
        args,
        train_loader=None,
        test_loader=None,
        class_idxs=None,
        **kwargs
    ):
        self.idx = idx
        self.args = args
        self.test_loader = test_loader
        self.train_loader = train_loader

        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha
        self.num_data = len(self.train_loader)
        self.class_idxs = class_idxs
        self.elapsed_comm_rounds = 0

        self.accuracies = []
        self.losses = []
        self.prune_rates = []
        self.cur_prune_rate = 0.00

        self.model = None
        self.global_model = None
        self.global_init_model = None

    def update(self) -> None:
        """
            Interface to Server
        """
        print(f"\n----------Client:{self.idx} Update---------------------")
        print(f'----------User Class ids: {self.class_idxs}------------')
        print(f"Evaluating Global model ")
        metrics = self.eval(self.global_model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model accuracy: {accuracy}')

        prune_summmary, num_zeros, num_global = get_prune_summary(model=self.global_model,
                                                                  name='weight')
        prune_rate = num_zeros / num_global
        print('Global model prune percentage: {}'.format(prune_rate))

        if self.cur_prune_rate < self.args.prune_threshold:
            if accuracy > self.eita:
                self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
                                          self.args.prune_threshold)
                if self.cur_prune_rate > prune_rate:
                    l1_prune(model=self.global_model,
                             amount=self.cur_prune_rate,
                             name='weight',
                             verbose=self.args.prune_verbose)
                    # reinitialize model with init_params
                    source_params = dict(
                        self.global_init_model.named_parameters())
                    for name, param in self.global_model.named_parameters():
                        param.data.copy_(source_params[name].data)
                    self.prune_rates.append(self.cur_prune_rate)
                else:
                    # reprune by the downloaded global-model(important)
                    # REVIEW: Rather than pruning each layer by orig_global_pruned_%,
                    # pruned each layer by its' orig_pruned_%
                    params_to_prune = get_prune_params(self.global_model)
                    for param, name in params_to_prune:
                        amount = torch.eq(getattr(param, name),
                                          0.00).sum().float()
                        prune.l1_unstructured(param, name, amount=int(amount))
                    self.prune_rates.append(prune_rate)

                self.model = self.global_model
                self.eita = self.eita_hat

            else:
                # reprune by the downloaded global-model(important)
                # REVIEW: Rather than pruning each layer by orig_global_pruned_%,
                # pruned each layer by its' orig_pruned_%
                params_to_prune = get_prune_params(self.global_model)
                for param, name in params_to_prune:
                    amount = torch.eq(getattr(param, name), 0.00).sum().float()
                    prune.l1_unstructured(param, name, amount=int(amount))
                self.eita *= self.alpha
                self.model = self.global_model
                self.prune_rates.append(prune_rate)
        else:
            if self.cur_prune_rate > prune_rate:
                l1_prune(model=self.global_model,
                         amount=self.cur_prune_rate,
                         name='weight',
                         verbose=self.args.prune_verbose)
                source_params = dict(self.global_init_model.named_parameters())
                for name, param in self.global_model.named_parameters():
                    param.data.copy_(source_params[name].data)
                self.prune_rates.append(self.cur_prune_rate)
            else:
                # reprune by the downloaded global-model(not important)
                params_to_prune = get_prune_params(self.global_model)
                for param, name in params_to_prune:
                    amount = torch.eq(getattr(param, name), 0.00).sum().float()
                    prune.l1_unstructured(param, name, amount=int(amount))
                self.prune_rates.append(prune_rate)

            self.model = self.global_model

        print(f"\nTraining local model")
        self.train(self.elapsed_comm_rounds)

        print(f"\nEvaluating Trained Model")
        metrics = self.eval(self.model)
        print(f'Trained model accuracy: {metrics["Accuracy"][0]}')

        wandb.log({f"{self.idx}_cur_prune_rate": self.cur_prune_rate})
        wandb.log({f"{self.idx}_eita": self.eita})
        wandb.log(
            {f"{self.idx}_percent_pruned": self.prune_rates[-1]})

        for key, thing in metrics.items():
            if(isinstance(thing, list)):
                wandb.log({f"{self.idx}_{key}": thing[0]})
            else:
                wandb.log({f"{self.idx}_{key}": thing})

        self.save(self.model)
        self.elapsed_comm_rounds += 1

    def train(self, round_index):
        """
            Train NN
        """
        losses = []

        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(
                    f"Client={self.idx}, epoch={epoch}, round:{round_index}")

            metrics = util_train(self.model,
                                 self.train_loader,
                                 self.args.lr,
                                 self.args.device,
                                 self.args.fast_dev_run,
                                 self.args.train_verbose)
            losses.append(metrics['Loss'][0])

            if self.args.fast_dev_run:
                break
        self.losses.extend(losses)

    @torch.no_grad()
    def download(self, global_model, global_init_model, *args, **kwargs):
        """
            Download global model from server
        """
        self.global_model = global_model
        self.global_init_model = global_init_model

        # params_to_prune = get_prune_params(self.global_model)
        # for param, name in params_to_prune:
        #     weights = getattr(param, name)
        #     masked = torch.eq(weights.data, 0.00).sum().item()
        #     # masked = 0.00
        #     prune.l1_unstructured(param, name, amount=int(masked))

        params_to_prune = get_prune_params(self.global_init_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            # masked = torch.eq(weights.data, 0.00).sum().item()
            masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

    def eval(self, model):
        """
            Eval self.model
        """
        eval_score = util_test(model,
                               self.test_loader,
                               self.args.device,
                               self.args.fast_dev_run,
                               self.args.test_verbose)
        self.accuracies.append(eval_score['Accuracy'][0])
        return eval_score

    def save(self, model, **kwargs):
        torch.save(model.state_dict(),
                   f"./checkpoints/c{self.idx}_model_{self.elapsed_comm_rounds}")

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        upload_model = copy_model(model=self.model, device=self.args.device)
        params_pruned = get_prune_params(upload_model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
        return {
            'model': upload_model,
            'acc': self.accuracies[-1]
        }
