from client import Client
from server import Server
# import datetime
# import time
# import os
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from utils import create_model
# import torch
# import random

# RANDOM_SEED = 42
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


# def random_seed(seed_value, use_cuda):
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     random.seed(seed_value)
#     if use_cuda:
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


# random_seed(RANDOM_SEED, True)


# def run_experiment(args, overrides):
#     for  k, v in overrides.items():
#         setattr(args, k, v)
#     args.log_folder = overrides['log_folder'] + '/' + overrides['exp_name']
#     print("Started getting data")
#     (client_loaders, val_loaders, test_loader), global_test_loader =\
#         get_data(args.num_clients,
#                  args.dataset, mode=args.data_split, batch_size=args.batch_size,
#                  num_train_samples_perclass = args.n_samples, n_class = args.n_class, rate_unbalance=args.rate_unbalance)
#     print("Finished getting data")
#     clients = []
#     print("Initializing clients")
#     for i in range(args.num_clients):
#         print("Client " + str(i))
#         clients.append(args.client(args, client_loaders[i], test_loader[i], client_id=i))
    
#     server = args.server(args, np.array(clients, dtype=np.object), test_loader=global_test_loader)
#     print("Now running the algorithm")
#     server.test_server_update() ##important
#     return server, clients


