import os
import sys
import errno
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as skmetrics
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
import torch.nn.utils.prune as prune
import io
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from torch.nn import functional as F
import gzip


@torch.no_grad()
def fed_avg(models: List[nn.Module], weights: torch.Tensor, device='cuda:0'):
    """
        models: list of nn.modules(unpruned/pruning removed)
        weights: normalized weights for each model
        cls:  Class of original model
    """
    aggr_model = models[0].__class__().to(device)
    model_params = []
    num_models = len(models)
    for model in models:
        model_params.append(dict(model.named_parameters()))

    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        for i in range(num_models):
            weighted_param = torch.mul(
                model_params[i][name].data, weights[i])
            param.data.copy_(param.data + weighted_param)
    return aggr_model


def create_model(cls, device='cuda:0') -> nn.Module:
    """
        Returns new model pruned by 0.00 %. This is necessary to create buffer masks
    """
    model = cls().to(device)
    l1_prune(model, amount=0.00, name='weight', verbose=False)
    return model


def copy_model(model: nn.Module, device='cuda:0'):
    """
        Returns a copy of the input model.
        Note: the model should have been pruned for this method to work to create buffer masks and what not.
    """
    new_model = create_model(model.__class__, device)
    source_params = dict(model.named_parameters())
    source_buffer = dict(model.named_buffers())
    for name, param in new_model.named_parameters():
        param.data.copy_(source_params[name].data)
    for name, buffer_ in new_model.named_buffers():
        buffer_.data.copy_(source_buffer[name].data)
    return new_model


metrics = MetricCollection([
    Accuracy(),
    Precision(),
    Recall(),
    F1(),
])


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    lr: float = 1e-3,
    device: str = 'cuda:0',
    fast_dev_run=False,
    verbose=True
) -> Dict[str, torch.Tensor]:

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    num_batch = len(train_dataloader)
    global metrics

    metrics = metrics.to(device)
    model.train(True)
    torch.set_grad_enabled(True)

    losses = []
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=num_batch,
                        disable=not verbose,
                        )

    for batch_idx, batch in progress_bar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        model.zero_grad()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        output = metrics(y_hat, y)

        progress_bar.set_postfix({'loss': loss.item(),
                                  'acc': output['Accuracy'].item()})
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    outputs = {k: [v.item()] for k, v in outputs.items()}
    torch.set_grad_enabled(False)
    outputs['Loss'] = [sum(losses) / len(losses)]
    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


# def evaluate(model, data_loader, verbose=True):
#     # Swithicing off gradient calculation to save memory
#     torch.no_grad()
#     # Switch to eval mode so that layers like Dropout function correctly
#     model.eval()

#     metric_names = ['Loss',
#                     'Accuracy',
#                     'Balanced Accuracy',
#                     'Precision Micro',
#                     'Recall Micro',
#                     'Precision Macro',
#                     'Recall Macro']

#     score = {name: [] for name in metric_names}

#     num_batch = len(data_loader)

#     progress_bar = tqdm(enumerate(data_loader),
#                         total=num_batch,
#                         file=sys.stdout)

#     for i, (x, ytrue) in progress_bar:

#         yraw = model(x)

#         _, ypred = torch.max(yraw, 1)

#         score = calculate_metrics(score, ytrue, yraw, ypred)

#         progress_bar.set_description('Evaluating')

#     for k, v in score.items():
#         score[k] = [sum(v) / len(v)]

#     if verbose:
#         print('Evaluation Score: ')
#         print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
#     model.train()
#     torch.enable_grad()
#     return score


# @torch.no_grad()
# def fevaluate(model, data_loader, verbose=True):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Switch to eval mode so that layers like Dropout function correctly
#     model.eval()
#     metric_names = ['Loss',
#                     'Accuracy',
#                     'Balanced Accuracy',
#                     'Precision Micro',
#                     'Recall Micro',
#                     'Precision Macro',
#                     'Recall Macro']

#     score = {name: [] for name in metric_names}

#     num_batch = len(data_loader)

#     classtypes = set()
#     progress_bar = tqdm(enumerate(data_loader),
#                         total=num_batch,
#                         file=sys.stdout)

#     for i, (x, ytrue) in progress_bar:
#         classtypes.add(int(ytrue[0]))
#         x = x.to(device)
#         # ytrue = ytrue.to(device)
#         yraw = model(x)
#         # yraw = yraw.to(device)
#         _, ypred = torch.max(yraw, 1)
#         # ypred = ypred.to(device)

#         score = calculate_metrics(score, ytrue, yraw.cpu(), ypred.cpu())

#         progress_bar.set_description('Evaluating')

#     pclass2 = ''
#     class_name = ['airplane', 'car', 'bird', 'cat',
#                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     for c in classtypes:
#         pclass2 += class_name[c]+' '
#     """
#     pclass = ''
#     for c in classtypes:
#         if c == 0:
#             pclass = pclass + 'airplane '
#         elif c == 1:
#             pclass = pclass + 'car '
#         elif c == 2:
#             pclass = pclass + 'bird '
#         elif c == 3:
#             pclass = pclass + 'cat '
#         elif c == 4:
#             pclass = pclass + 'deer '
#         elif c == 5:
#             pclass = pclass + 'dog '
#         elif c == 6:
#             pclass = pclass + 'frog '
#         elif c == 7:
#             pclass = pclass + 'horse '
#         elif c == 8:
#             pclass = pclass + 'ship '
#         elif c == 9:
#             pclass = pclass + 'truck '
#             """
#     for k, v in score.items():
#         score[k] = [sum(v) / len(v)]

#     print('Acc. for classes', classtypes, pclass2,
#           ": ", score['Accuracy'][-1], flush=True)

#     if verbose:
#         print('Evaluation Score: ')
#         print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
#     model.train()
#     torch.enable_grad()
#     return score


@ torch.no_grad()
def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    device='cuda:0',
    fast_dev_run=False,
    verbose=True,
) -> Dict[str, torch.Tensor]:

    num_batch = len(test_dataloader)
    model.eval()
    global metrics

    metrics = metrics.to(device)
    progress_bar = tqdm(enumerate(test_dataloader),
                        total=num_batch,
                        file=sys.stdout,
                        disable=not verbose)
    for batch_idx, batch in progress_bar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        output = metrics(y_hat, y)

        progress_bar.set_postfix({'acc': output['Accuracy'].item()})
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    model.train(True)
    outputs = {k: [v.item()] for k, v in outputs.items()}

    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


def l1_prune(model, amount=0.00, name='weight', verbose=False, glob=False):
    """
        Prunes the model param by param by given amount
    """
    params_to_prune = get_prune_params(model, name)
    if glob:
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount)
    else:
        for params, name in params_to_prune:
            prune.l1_unstructured(params, name, amount)
    if verbose:
        info, _, _ = get_prune_summary(model, name)
        global_pruning = info['global']
        info.pop('global')
        print(tabulate(info, headers='keys', tablefmt='github'))
        print("Total Pruning: {}%".format(global_pruning))


"""
Hadamard Mult of Mask and Attributes,
then return zeros
"""


@ torch.no_grad()
def summarize_prune(model: nn.Module, name: str = 'weight') -> tuple:
    """
        returns (pruned_params,total_params)
    """
    num_pruned = 0
    params, num_global_weights, _ = get_prune_params(model)
    for param, _ in params:
        if hasattr(param, name+'_mask'):
            data = getattr(param, name+'_mask')
            num_pruned += int(torch.sum(data == 0.0).item())
    return (num_pruned, num_global_weights)


def get_prune_params(model, name='weight') -> List[Tuple[nn.Parameter, str]]:
    params_to_prune = []
    for _, module in model.named_children():
        for name_, param in module.named_parameters():
            if name in name_:
                params_to_prune.append((module, name))
    return params_to_prune


def get_prune_summary(model, name='weight') -> List[Union[Union[Dict[str, Union[List[Union[str, float]], float]], int], int]]:
    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    num_global_weights = 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}
    params_pruned = get_prune_params(model, 'weight')

    for layer, weight_name in params_pruned:

        num_layer_zeros = torch.sum(
            getattr(layer, weight_name) == 0.0).item()
        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        num_global_weights += num_layer_weights
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(layer.__str__())
        prune_stat['Weight Name'].append(weight_name)
        prune_stat['Percent Pruned'].append(
            f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

    global_prune_percent = num_global_zeros / num_global_weights

    prune_stat['global'] = global_prune_percent
    return prune_stat, num_global_zeros, num_global_weights


def custom_save(model, path):
    """
    https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
    Custom save utility function
    Compresses the model using gzip
    Helpfull if model is highly pruned
    """
    bufferIn = io.BytesIO()
    torch.save(model.state_dict(), bufferIn)
    bufferOut = gzip.compress(bufferIn.getvalue())
    with gzip.open(path, 'wb') as f:
        f.write(bufferOut)


def custom_load(path) -> Dict:
    """
    returns saved_dictionary
    """
    with gzip.open(path, 'rb') as f:
        bufferIn = f.read()
        bufferOut = gzip.decompress(bufferIn)
        state_dict = torch.load(io.BytesIO(bufferOut))
    return state_dict


def log_obj(path, obj):
    # pass
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #
    with open(path, 'wb') as file:
        if isinstance(obj, nn.Module):
            torch.save(obj, file)
        else:
            pickle.dump(obj, file)


class CustomPruneMethod(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount, orig_weights):
        super().__init__()
        self.amount = amount
        self.original_signs = self.get_signs_from_tensor(orig_weights)

    def get_signs_from_tensor(self, t: torch.Tensor):
        return torch.sign(t).view(-1)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        large_weight_mask = t.view(-1).mul(self.original_signs)
        large_weight_mask_ranked = F.relu(large_weight_mask)
        nparams_toprune = int(torch.numel(t) * self.amount)  # get this val
        if nparams_toprune > 0:
            bottom_k = torch.topk(
                large_weight_mask_ranked.view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[bottom_k.indices] = 0.00
            return mask
        else:
            return mask


def customPrune(module, orig_module, amount=0.1, name='weight'):
    """
        Taken from https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        Takes: current module (module), name of the parameter to prune (name)

    """
    CustomPruneMethod.apply(module, name, amount, orig_module)
    return module


def super_prune(
    model: nn.Module,
    init_model: nn.Module,
    amount: float = 0.0,
    name: str = 'weight'
) -> None:
    """

    """
    params_to_prune = get_prune_params(model)
    init_params = get_prune_params(init_model)

    for idx, (param, name) in enumerate(params_to_prune):
        orig_params = getattr(init_params[idx][0], name)

        # original params are sliced by the pruned model's mask
        # this is because pytorch's pruning container slices the mask by
        # non-zero params
        if hasattr(param, 'weight_mask'):
            mask = getattr(param, 'weight_mask')
            sliced_params = orig_params[mask.to(torch.bool)]
            customPrune(param, sliced_params, amount, name)
        else:
            customPrune(param, orig_params, amount, name)
