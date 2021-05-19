import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from utils import copy_model, create_model, get_prune_summary, train, ftrain, \
    evaluate, fevaluate, train, ftrain, evaluate, fevaluate, fprune_fixed_amount,\
    prune_fixed_amount, copy_model, create_model, get_prune_summary, log_obj,summarize_prune
import numpy as np
from typing import Dict
import copy
import math
import wandb

torch.manual_seed(42)
np.random.seed(42)


class Client():
    def __init__(self, args, train_loader, test_loader, client_id=None):

        self.args = args
        print("Creating model for client " + str(client_id))
        self.model = create_model(self.args.dataset, self.args.arch)
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.losses = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.prune_rates = np.zeros(args.comm_rounds)
        assert self.model, "Something went wrong and the model cannot be initialized"
        #######

    def update(self) -> None:
        """
            Interface to Server
        """
        print(f"----------Client:{self.client_id} Update---------------------")

        # evaluate globalModel on local data
        # if accuracy < eita proceed as straggler else LH finder
        eval_score = self.eval(self.globalModel)
            
        # get pruning summary for globalModel
        num_pruned, num_params = summarize_prune(
            self.globalModel, name='weight')
        cur_prune_rate = num_pruned / num_params

        if eval_score["Accuracy"][0] > self.args.eita:
            #--------------------Lottery Finder-----------------#



            # expected final pruning % of local model
            # prune model by prune_rate - current_prune_rate
            # every iteration pruning should be increase by prune_step if viable
            prune_rate = min(cur_prune_rate + self.args.prune_step,
                             self.args.prune_percent)
            self.prune(self.globalModel,
                       prune_rate=prune_rate - cur_prune_rate)
            self.prune_rates[self.elapsed_comm_rounds] = prune_rate
            # reinit the model by global_initial_model params
            # TODO: check reinit function
            self.model = copy_model(self.global_initModel,
                                    self.args.dataset,
                                    self.args.arch,
                                    source_buff= dict(self.globalModel.named_buffers()))

            # eita reinitialized to original val
            self.args.eita = self.args.eita_hat

        else:
            #---------------------Straggler-----------------------------#
            # eita = eita*alpha
            self.args.eita *= self.args.alpha
            self.prune_rates[self.elapsed_comm_rounds] = cur_prune_rate
            # copy globalModel
            self.model = self.globalModel

        #-----------------------TRAINING LOOOP ------------------------#
        # train both straggler and LH finder
        self.model.train()
        self.train(self.elapsed_comm_rounds)

        self.eval_score = self.eval(self.model)

        wandb.log({f"{self.client_id}_cur_prune_rate": cur_prune_rate})
        wandb.log({f"{self.client_id}_eita": self.args.eita})

        for key, thing in self.eval_score.items():
            if(isinstance(thing, list)):
                wandb.log({f"{self.client_id}_{key}": thing[0]})
            else:
                wandb.log({f"{self.client_id}_{key}": thing.item()})

        if (self.elapsed_comm_rounds+1) % self.args.save_freq == 0:
            self.save(self.model)

        self.elapsed_comm_rounds += 1

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

            if self.args.report_verbosity:
                epoch_path = train_log_path + f'client_model_epoch{epoch}.torch'
                epoch_score_path = train_log_path + \
                    f'client_train_score_epoch{epoch}.pickle'
                log_obj(epoch_path, self.model)
                log_obj(epoch_score_path, train_score)

        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)

    def prune(self, model, prune_rate, *args, **kwargs):
        """
            Prune model
        """
        fprune_fixed_amount(model, prune_rate,  # prune_step,
                            verbose=self.args.prune_verbosity)

    def download(self, globalModel, global_initModel, *args, **kwargs):
        """
            Download global model from server
        """
        self.globalModel = globalModel
        if global_initModel is not None:
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
        if self.args.report_verbosity:
            eval_log_path1 = f"./log/full_save/client{self.client_id}/round{self.elapsed_comm_rounds}_model.pickle"
            eval_log_path2 = f"./log/full_save/client{self.client_id}/round{self.elapsed_comm_rounds}_dict.pickle"
            log_obj(eval_log_path1, self.model)
            log_obj(eval_log_path2, self.__dict__)

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
