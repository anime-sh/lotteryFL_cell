import os
import torch
import argparse
import pickle
from pytorch_lightning import seed_everything
from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP
from server import Server
from client import Client
from util import create_model
import wandb
from dataset.datasource import DataLoaders
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1

models = {
    'cifar10': {
        'cnn': CIFAR_CNN,
        'mlp': CIFAR_MLP
    },
    'mnist': {
        'cnn': MNIST_CNN,
        'mlp': MNIST_MLP
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="mnist|cifar10",
                        type=str, default="cifar10")
    parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
    parser.add_argument('--dataset_mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--rate_unbalance', type=float, default=1.0)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--prune_step', type=float, default=0.2)
    parser.add_argument('--prune_threshold', type=float, default=0.6)
    parser.add_argument('--server_prune', type=bool, default=False)
    parser.add_argument('--server_prune_step', type=float, default=0.2)
    parser.add_argument('--server_prune_freq', type=int, default=10)
    parser.add_argument('--frac_clients_per_round', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--eita', type=float, default=0.5,
                        help="accuracy threshold")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="accuracy reduction factor")
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--train_verbose', type=bool, default=False)
    parser.add_argument('--test_verbose', type=bool, default=False)
    parser.add_argument('--prune_verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    seed_everything(seed=args.seed, workers=True)

    model = create_model(cls=models[args.dataset]
                         [args.arch], device=args.device)

    train_loaders, test_loaders, class_idxs = DataLoaders(num_users=args.num_clients,
                                                          dataset_name=args.dataset,
                                                          n_class=args.n_class,
                                                          nsamples=args.n_samples,
                                                          mode=args.dataset_mode,
                                                          batch_size=args.batch_size,
                                                          rate_unbalance=args.rate_unbalance,
                                                          num_workers=args.num_workers)
    clients = []
    for i in range(args.num_clients):
        client = Client(
            i, args, train_loaders[i], test_loaders[i], class_idxs[i])
        clients.append(client)

    wandb.login()
    wandb.init(project="Test1")
    wandb.config.update(args)

    server = Server(args, model, clients)

    for i in range(args.rounds):
        server.update()
