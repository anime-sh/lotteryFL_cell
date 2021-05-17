import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
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
        # print("Copying model for client " + str(client_id))
        # self.init_model = copy_model(
        #     self.model, self.args.dataset, self.args.arch)
        # print("Done Copying model for client " + str(client_id))
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.losses = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.prune_rates = np.zeros(args.comm_rounds)
        assert self.model, "Something went wrong and the model cannot be initialized"
        #######

    def update(self, round_index) -> None:
        """
            Interface to Server
        """
        print("--------------STARTED UPDATE----------")
        self.elapsed_comm_rounds += 1
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params

        eval_score = self.eval(self.globalModel)

        if eval_score["Accuracy"][0] > self.args.eita:
            prune_rate = min(cur_prune_rate + self.args.prune_step,
                             self.args.prune_percent)

            self.model = copy_model(self.global_initModel,
                                    self.args.dataset,
                                    self.args.arch,
                                    dict(self.globalModel.named_buffers()))
            self.prune(self.model, prune_rate)
            self.args.eita = self.args.eita_hat

        else:
            self.args.eita *= self.args.alpha
            self.model = copy_model(self.globalModel,
                                    self.args.dataset,
                                    self.args.arch)

        #-----------------------TRAINING LOOOP ------------------------#
        self.train(round_index)
        self.eval_score = self.eval(self.model)
        self.save(self.model)

    def train(self, round_index):
        """
            Train NN
        """
        accuracies = []
        losses = []

        for epoch in range(self.args.client_epoch):
            train_log_path = f'./log/clients/client{self.client_id}'\
                             f'/round{self.elapsed_comm_rounds}/'
            if self.args.train_verbosity:
                print(f"Client={self.client_id}, epoch={epoch}")
            train_score = ftrain(self.model,
                                 self.train_loader,
                                 self.args.lr,
                                 self.args.train_verbosity)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
            epoch_path = train_log_path + f'client_model_epoch{epoch}.torch'
            epoch_score_path = train_log_path + \
                f'client_train_score_epoch{epoch}.pickle'
            log_obj(epoch_path, self.model)
            log_obj(epoch_score_path, train_score)
    
        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)

    def prune(self, model, prune_rate, *args, **kwargs):
        """
            Prune self.model
        """
        fprune_fixed_amount(model, prune_rate,  # prune_step,
                            verbose=self.args.prune_verbosity)

    def download(self, globalModel, global_initModel, *args, **kwargs):
        """
            Download global model from server
        """
        self.globalModel = globalModel
        self.global_initModel = global_initModel

    def eval(self, model):
        """
            Eval self.model
        """
        eval_score = fevaluate(model,
                               self.test_loader,
                               verbose=self.args.test_verbosity)
        if self.args.test_verbosity:
            eval_log_path = f'./log/clients/client{self.client_id}/'\
                            f'round{self.elapsed_comm_rounds}/'\
                            f'eval_score_round{self.elapsed_comm_rounds}.pickle'
            log_obj(eval_log_path, eval_score)
        return eval_score

    def save(self, *args, **kwargs):
        """
            Save model,meta-info,states
        """
        eval_log_path1 = f"./log/full_save/client{self.client_id}/round{self.elapsed_comm_rounds}_model.pickle"
        eval_log_path2 = f"./log/full_save/client{self.client_id}/round{self.elapsed_comm_rounds}_dict.pickle"
        if self.args.verbose:
            log_obj(eval_log_path1,self.model)
            log_obj(eval_log_path2,self.__dict__)

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
