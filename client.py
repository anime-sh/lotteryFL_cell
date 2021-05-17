import torch
import torch.nn as nn
import torch.optim as optim
from torch import Module
import numpy as np
import os
from utils import copy_model, create_model, get_prune_summary, train, ftrain, evaluate, fevaluate, train, ftrain, evaluate, fevaluate, fprune_fixed_amount, prune_fixed_amount, copy_model, create_model, get_prune_summary, log_obj
import numpy as np
from typing import Dict
import copy
import math

# from util import train, ftrain, evaluate, fevaluate, fprune_fixed_amount, fprune_fixed_amount_res, prune_fixed_amount, copy_model, \
# #   create_model, get_prune_summary, log_obj, merge_models, get_prune_summary_res


torch.manual_seed(42)
np.random.seed(42)


class Client():
    def __init__(self, args, train_loader, test_loader, client_id=None):

        self.args = args
        print("Creating model for client " + str(client_id))
        self.model = create_model(self.args.dataset, self.args.arch)
        print("Copying model for client " + str(client_id))
        self.init_model = copy_model(
            self.model, self.args.dataset, self.args.arch)
        print("Done Copying model for client " + str(client_id))
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.losses = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.prune_rates = np.zeros(args.comm_rounds)
        assert self.model, "Something went wrong and the model cannot be initialized"
        #######

    def update(self, global_model, global_init_model, round_index) -> None:
        """
            Interface to Server
        """
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params

        self.eval_score = self.eval(self.globalModel)

        if self.eval_score["Accuracy"][0] > self.args.eita:
            prune_rate = min(cur_prune_rate + self.args.prune_step,
                             self.args.prune_percent)
            self.prune(self.globalmodel, prune_rate)
            self.model = copy_model(self.global_initModel,
                                    self.args.dataset,
                                    self.args.arch,
                                    dict(self.globalModel.named_buffers()))
            self.args.eita = self.args.eita_hat
        else:
            self.args.eita *= self.args.alpha
            self.model = copy_model(self.globalModel,
                                    self.args.dataset,
                                    self.args.arch)

        #-----------------------TRAINING LOOOP ------------------------#
        train_score = self.train()
        self.save(self.model)

    def train(self):
        """
            Train NN
        """
        for i in range(self.args.client_epoch):
            train_log_path = f'./log/clients/client{self.client_id}'\
                             f'/round{self.elapsed_comm_rounds}/'
            # print(f'Epoch {i+1}')
            train_score = ftrain(self.model,
                                self.train_load,
                                
                  verbose=self.ar
            accuracies.append(train_score['Accuracy'][-1])
            epoch_path = train_log_path + f'client_model_epoch{i}.torch'
            epoch_score_path = train_log_path + \
                f'client_train_score_epoch{i}.pickle'
            log_obj(epoch_path, self.model)

            if self.args.verboses:
                print(f"Client={client_id},Epoch= {epoch}")
                print(tabulate(average_scores, headers='keys', tablefmt='github'))            log_obj(epoch_score_path, train_score)

        pass

    def prune(self, model, prune_rate, *args, **kwargs):
        """
            Prune self.model
        """
        pass

    def download(self, globalModel, global_initModel, *args, **kwargs):
        """
            Download global model from server
        """
        self.globalModel = globalModel
        self.global_initModel = global_initModel

    def eval(self, *args, **kwargss):
        """
            Eval self.model
        """
        eval_flag = 0
        eval_score = fevaluate(self.model,
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        fprune_fixed_amount(self.model,
                               0,  # prune_step,
                               verbose=self.args.prune_verbosity)

        eval_log_path = f'./log/clients/client{self.client_id}/'\
                        f'round{self.elapsed_comm_rounds}/'\
                        f'eval_score_round{self.elapsed_comm_rounds}.pickle'
        log_obj(eval_log_path, eval_score)
        return eval_score

    def save(self, *args, **kwargs):
        """
            Save model,meta-info,states
        """
        pass

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        return {"model": copy_model(self.model,
                                    self.args.dataset,
                                    self.args.dataset),
                                    self.configs.dataset,
                                    self.configs.dataset),
                "acc": self.eval_score["Accuracy"]
                }
