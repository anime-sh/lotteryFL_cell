from client import Client
from server import Server
import datetime
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import run_experiments
import torch
import random
from provided_code.datasource import get_data

RANDOM_SEED = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


random_seed(RANDOM_SEED, True)


def build_args(arch='mlp',
               dataset='mnist',
               data_split='non-iid',
               client=Client,
               server=Server,
               n_class=2,
               n_samples=20,
               rate_unbalance=1,
               avg_logic=None,
               num_clients=10,
               comm_rounds=10,
               frac=0.3,
               prune_step=0.15,
               prune_percent=0.45,
               acc_thresh=0.5,
               client_epoch=10,
               batch_size=4,
               lr=0.001,
               eita_hat=0.5,
               eita=0.5,
               alpha=0.75,
               train_verbosity=True,
               test_verbosity=True,
               prune_verbosity=True,
               prune_threshold=0.5,
               globalPrune = False,
               report_verbosity=True
               ):

    args = type('', (), {})()
    args.arch = arch
    args.dataset = dataset
    args.data_split = data_split
    args.client = client
    args.server = server
    args.num_clients = num_clients
    args.lr = lr
    args.batch_size = batch_size
    args.comm_rounds = comm_rounds
    args.frac = frac
    args.client_epoch = client_epoch
    args.acc_thresh = acc_thresh
    args.prune_percent = prune_percent
    args.prune_step = prune_step
    args.train_verbosity = train_verbosity
    args.test_verbosity = test_verbosity
    args.prune_verbosity = prune_verbosity
    args.avg_logic = avg_logic
    args.n_class = n_class
    args.n_samples = n_samples
    args.rate_unbalance = rate_unbalance
    args.prune_threshold = prune_threshold
    args.eita = eita
    args.eita_hat = eita_hat
    args.alpha = alpha
    args.globalPrune = globalPrune
    args.report_verbosity = report_verbosity
    return args


CIFAR10_experiments = {
    'CIFAR10_with_server_side_pruning':
        build_args(arch='cnn',
                   client=Client,
                   server=Server,
                   dataset='cifar10',
                   avg_logic='standalone',
                   num_clients=5,
                   comm_rounds=400,
                   frac=1,
                   prune_step=0.2,
                   prune_percent=0.8,
                   acc_thresh=0.5,
                   client_epoch=10,
                   batch_size=32,
                   lr=0.001,
                   rate_unbalance=1.0,
                   n_samples=50,
                   n_class=4,
                   eita_hat=0.5,
                   eita=0.5,
                   alpha=0.75,
                   prune_threshold=0.0125,
                   report_verbosity=False,
                   train_verbosity=True,
                   test_verbosity=True,
                   globalPrune = True,
                   prune_verbosity=True)
}

if __name__ == "__main__":
    overrides = {
        'log_folder': './report_output',
        'running_on_cloud': False
    }
    run_experiments(CIFAR10_experiments, overrides)
