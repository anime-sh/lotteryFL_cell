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
import torch.nn.utils.prune as prune

torch.manual_seed(0)
np.random.seed(0)


def fed_avg(models, dataset, arch, data_nums):
    print("IN FED_AVG MODE\n", flush=True)
    # copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    new_model = create_model(dataset, arch)
    num_models = len(models)
    num_data_total = sum(data_nums)
    data_nums = data_nums / num_data_total
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                weighted_param = torch.mul(weights[i][name], data_nums[i])
                param.data.copy_(param.data + weighted_param)
        avg = torch.div(param.data, num_models)
        param.data.copy_(avg)
    return new_model


def merge_models(models, dataset, arch, tally, round_index):
    print("MERGING MODELS\n", flush=True)
    # copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    new_model = create_model(dataset, arch)
    #new_model = copy_model(models[round_index], dataset, arch)
    num_models = len(models)
    #num_data_total = sum(data_nums)
    #data_nums = data_nums / num_data_total
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))

        # for name, param in new_model.named_parameters():
        #    param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        tallysum = 0
        for name, param in new_model.named_parameters():
            param.data.copy_(
                torch.mul(weights[round_index][name], round_index+1))
            tallysum = 0
            for i in range(round_index-1):
                if tally[i] != 0.0:
                    #weighted_param = torch.mul(weights[i][name], data_nums[i])
                    weighted_param = torch.mul(weights[i][name], i+1)
                    param.data.copy_(param.data + weighted_param)
                    tallysum += (i+1)
        print(sum(tally)+1)
        #tallysum /= 5
        avg = torch.div(param.data, tallysum+round_index+1)
        param.data.copy_(avg)
    return new_model


def lottery_fl_v2(server_model, models, dataset, arch, data_nums):
    print("IN LOT FLv2 MODE\n", flush=True)
    # copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    new_model = create_model(dataset, arch)
    num_models = len(models)
    num_data_total = sum(data_nums)
    data_nums = data_nums / num_data_total
    with torch.no_grad():
        # Get server weights
        server_weights = dict(server_model.named_parameters())

        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                #parameters_to_prune, num_global_weights, _ = get_prune_params(models[i])

                model_masks = masks[i]

                try:
                    layer_mask = model_masks[name.strip("_orig") + "_mask"]
                    weights[i][name] *= layer_mask
                    weights[i][name] = np.where(
                        weights[i][name] != 0, weights[i][name], server_weights[name])
                except Exception as e:
                    # print("exceptions")
                    # print(e)
                    pass
                weighted_param = weights[i][name]
                #weighted_param = torch.mul(weights[i][name], data_nums[i])
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            param.data.copy_(avg)
    return new_model


def lottery_fl_v3(server_model, models, dataset, arch, data_nums):
    print("IN LOT FLv3 MODE\n", flush=True)
    # copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    new_model = create_model(dataset, arch)
    num_models = len(models)
    num_data_total = sum(data_nums)
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            new_c_model = copy_model(models[i], dataset, arch)
            parameters_to_prune, _, _ = get_prune_params(new_c_model)
            for m, n in parameters_to_prune:
                prune.remove(m, n)
            weights.append(dict(new_c_model.named_parameters()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                # torch.mul(weights[i][name], data_nums[i])
                weighted_param = weights[i][name.strip("_orig")]
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            param.data.copy_(avg)
    return new_model


def lottery_fl_v3_res(server_model, models, dataset, arch, data_nums):
    print("IN LOT FLv3 MODE\n", flush=True)
    # copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    new_model = create_model(dataset, arch)
    num_models = len(models)
    num_data_total = sum(data_nums)
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            new_c_model = copy_model(models[i], dataset, arch)
            parameters_to_prune, _, _ = get_prune_params_res(new_c_model)
            for m, n in parameters_to_prune:
                prune.remove(m, n)
            weights.append(dict(new_c_model.named_parameters()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                # torch.mul(weights[i][name], data_nums[i])
                weighted_param = weights[i][name.strip("_orig")]
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            param.data.copy_(avg)
    return new_model


def average_weights(models, dataset, arch, data_nums):
    print("IN LOT FLv1 MODE\n", flush=True)
    # copy_model(server_model, dataset, arch, source_buff=dict(server_model.named_buffers()))
    new_model = create_model(dataset, arch)
    num_models = len(models)
    num_data_total = sum(data_nums)
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))

        for name, param in new_model.named_parameters():
            param.data.copy_(torch.zeros_like(param.data))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(num_models):
                # torch.mul(weights[i][name], data_nums[i])
                weighted_param = weights[i][name]
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            param.data.copy_(avg)
    return new_model


def average_weights_masks(models, dataset, arch):
    print("IN MASkS MODE\n", flush=True)
    new_model = copy_model(models[0], dataset, arch)
    num_models = len(models)
    print(f"Averaging weights with num_models {num_models}", flush=True)
    with torch.no_grad():
        # Getting all the weights and masks from original models
        weights, masks = [], []
        for i in range(num_models):
            weights.append(dict(models[i].named_parameters()))
            masks.append(dict(models[i].named_buffers()))
        # Averaging weights
        for name, param in new_model.named_parameters():
            for i in range(1, num_models):
                # print(f"Client #{i} has weight for avg: { data_nums[i] } / {num_data_total} = { data_nums[i] / num_data_total}")
                # torch.mul(weights[i][name], data_nums[i])
                weighted_param = weights[i][name]
                param.data.copy_(param.data + weighted_param)
            avg = torch.div(param.data, num_models)
            #avg = torch.div(param.data, num_data_total)
            param.data.copy_(avg)
        # SET masks back to 1 for global
        for name, buffer in new_model.named_buffers():
            for i in range(1, num_models):
                # weighted_masks = torch.mul(masks[i][name], data_nums[i])
                weighted_masks = masks[i][name]
                buffer.data.copy_(buffer.data + weighted_masks)
             #torch.mul(weights[i][name], data_nums[i])
            #avg = torch.ones_like(buffer.data)

            # The code below clips all the values to [0.0, 1.0] of the new model.
            # This might seems trivial, but if you don't do this, you will get
            # an error message saying that there's not parameters to prune.
            # This has something to do with how pruning is handled internally

            # avg = torch.div(buffer.data, num_data_total)
            avg = torch.div(buffer.data, num_models)
            avg = torch.ceil(avg)

            #avg = torch.clamp(avg, 0.0, 1.0)
            #avg = torch.round(avg)
            buffer.data.copy_(avg)
    return new_model


def copy_model(model, dataset, arch, source_buff=None):
    new_model = create_model(dataset, arch)
    source_weights = dict(model.named_parameters())
    source_buffers = source_buff if source_buff else dict(model.named_buffers())
    for name, param in new_model.named_parameters():
        param.data.copy_(source_weights[name])
    for name, buffer in new_model.named_buffers():
        buffer.data.copy_(source_buffers[name])

    #prune.random_unstructured(new_model, name="weight", amount=0.0)
    #prune_fixed_amount(new_model, 0, verbose=False)

    return new_model


def create_model(dataset_name, model_type) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "mnist":
        from model.mnist import mlp, cnn
        # print("MNIST")
    elif dataset_name == "cifar10":
        from model.cifar10 import mlp, cnn, resnet
        #print("Got CIFAR10")

    else:
        print("You did not enter the name of a supported architecture for this dataset")
        print("Supported datasets: {}, {}".format('"CIFAR10"', '"MNIST"'))
        exit()

    if model_type == 'mlp':
        new_model = mlp.MLP().to(device)
        # This pruning call is made so that the model is set up for accepting
        # weights from another pruned model. If this is not done, the weights
        # will be incompatible
        prune_fixed_amount(new_model, 0.0, verbose=False)
        return new_model

    elif model_type == 'cnn':
        new_model = cnn.CNN().to(device)
        prune_fixed_amount(new_model, 0.0, verbose=False)
        return new_model

    elif model_type == 'resnet':
        new_model = resnet.ResNet18().to(device)
        fprune_fixed_amount_res(new_model, 0.0, verbose=False)
        return new_model

    else:
        print("You did not enter the name of a supported architecture for this dataset")
        print("Supported models: {}, {}".format('"mlp"', '"cnn"'))
        exit()


def train(model,
          train_loader,
          lr=0.001,
          verbose=True):

    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    num_batch = len(train_loader)
    model.train()
    metric_names = ['Loss',
                    'Accuracy',
                    'Balanced Accuracy',
                    'Precision Micro',
                    'Recall Micro',
                    'Precision Macro',
                    'Recall Macro']

    score = {name: [] for name in metric_names}

    progress_bar = tqdm(enumerate(train_loader),
                        total=num_batch,
                        file=sys.stdout)
    # Iterating over all mini-batches
    for i, data in progress_bar:

        x, ytrue = data

        yraw = model(x)

        loss = loss_function(yraw, ytrue)

        model.zero_grad()

        loss.backward()

        opt.step()

        # Truning the raw output of the network into one-hot result
        _, ypred = torch.max(yraw, 1)

        score = calculate_metrics(score, ytrue, yraw, ypred)

    average_scores = {}

    for k, v in score.items():
        average_scores[k] = [sum(v) / len(v)]
        score[k].append(sum(v) / len(v))

    # if verbose:
        # print(f"round={round}, client={client_id}, epoch= {epoch}: ")
        # print(tabulate(average_scores, headers='keys', tablefmt='github'))

    return score

# train with freezing


def ftrain(model,
           train_loader,
           lr=0.001,
           verbose=True):
    for mod in list(model.modules()):
        if type(mod) != nn.Sequential and type(mod) != type(model):
            for name, param in mod.named_parameters():
                if 'weight' in name:
                    prune.remove(mod, 'weight')
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    #device = torch.device("cpu")
    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    num_batch = len(train_loader)
    model.train()
    metric_names = ['Loss', 'Accuracy', 'Balanced Accuracy',
                    'Precision Micro', 'Recall Micro', 'Precision Macro', 'Recall Macro']

    score = {name: [] for name in metric_names}

    progress_bar = tqdm(enumerate(train_loader),
                        total=num_batch,
                        file=sys.stdout)
    # Iterating over all mini-batches
    for i, data in progress_bar:
        x, ytrue = data[0].to(device), data[1].to(device)
        yraw = model(x)
        loss = loss_function(yraw, ytrue)
        model.zero_grad()
        loss.backward()

        # Truning the raw output of the network into one-hot result
        _, ypred = torch.max(yraw, 1)
        ypred.to(device)

        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(np.abs(tensor) < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        opt.step()

        # yraw.to("cpu")
        score = calculate_metrics(score, ytrue.cpu(), yraw.cpu(), ypred.cpu())

    average_scores = {}

    for k, v in score.items():
        average_scores[k] = [sum(v) / len(v)]
        score[k].append(sum(v) / len(v))

    if verbose:
        # print(f"round={round}, client={client_id}, epoch= {epoch}: ")
        print(tabulate(average_scores, headers='keys', tablefmt='github'))

    return score


def evaluate(model, data_loader, verbose=True):
    # Swithicing off gradient calculation to save memory
    torch.no_grad()
    # Switch to eval mode so that layers like Dropout function correctly
    model.eval()

    metric_names = ['Loss',
                    'Accuracy',
                    'Balanced Accuracy',
                    'Precision Micro',
                    'Recall Micro',
                    'Precision Macro',
                    'Recall Macro']

    score = {name: [] for name in metric_names}

    num_batch = len(data_loader)

    progress_bar = tqdm(enumerate(data_loader),
                        total=num_batch,
                        file=sys.stdout)

    for i, (x, ytrue) in progress_bar:

        yraw = model(x)

        _, ypred = torch.max(yraw, 1)

        score = calculate_metrics(score, ytrue, yraw, ypred)

        progress_bar.set_description('Evaluating')

    for k, v in score.items():
        score[k] = [sum(v) / len(v)]

    if verbose:
        print('Evaluation Score: ')
        print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
    model.train()
    torch.enable_grad()
    return score


def fevaluate(model, data_loader, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Swithicing off gradient calculation to save memory
    torch.no_grad()
    # Switch to eval mode so that layers like Dropout function correctly
    model.eval()

    metric_names = ['Loss',
                    'Accuracy',
                    'Balanced Accuracy',
                    'Precision Micro',
                    'Recall Micro',
                    'Precision Macro',
                    'Recall Macro']

    score = {name: [] for name in metric_names}

    num_batch = len(data_loader)

    classtypes = set()
    progress_bar = tqdm(enumerate(data_loader),
                        total=num_batch,
                        file=sys.stdout)

    for i, (x, ytrue) in progress_bar:
        classtypes.add(int(ytrue[0]))
        x = x.to(device)
        #ytrue = ytrue.to(device)
        yraw = model(x)
        #yraw = yraw.to(device)
        _, ypred = torch.max(yraw, 1)
        #ypred = ypred.to(device)

        score = calculate_metrics(score, ytrue, yraw.cpu(), ypred.cpu())

        progress_bar.set_description('Evaluating')

    pclass2 = ''
    class_name = ['airplane', 'car', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for c in classtypes:
        pclass2 += class_name[c]+' '
    """
    pclass = ''
    for c in classtypes:
        if c == 0:
            pclass = pclass + 'airplane '
        elif c == 1:
            pclass = pclass + 'car '
        elif c == 2:
            pclass = pclass + 'bird '
        elif c == 3:
            pclass = pclass + 'cat '
        elif c == 4:
            pclass = pclass + 'deer '
        elif c == 5:
            pclass = pclass + 'dog '
        elif c == 6:
            pclass = pclass + 'frog '
        elif c == 7:
            pclass = pclass + 'horse '
        elif c == 8:
            pclass = pclass + 'ship '
        elif c == 9:
            pclass = pclass + 'truck '
            """
    for k, v in score.items():
        score[k] = [sum(v) / len(v)]

    print('Acc. for classes', classtypes, pclass2,
          ": ", score['Accuracy'][-1], flush=True)

    if verbose:
        print('Evaluation Score: ')
        print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
    model.train()
    torch.enable_grad()
    return score


def prune_fixed_amount(model, amount, verbose=True, glob=True):
    parameters_to_prune, num_global_weights, layers_w_count = get_prune_params(
        model)

    if glob:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=math.floor(amount * num_global_weights))
    else:
        for i, (m, n) in enumerate(parameters_to_prune):
            prune.l1_unstructured(m, name=n, amount=math.floor(
                amount * layers_w_count[i][1]))

    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}

    # Pruning is done in-place, thus parameters_to_prune is updated
    for layer, weight_name in parameters_to_prune:

        num_layer_zeros = torch.sum(getattr(layer, weight_name) == 0.0).item()
        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(layer.__str__())
        prune_stat['Weight Name'].append(weight_name)
        prune_stat['Percent Pruned'].append(
            f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

    global_prune_percent = num_global_zeros / num_global_weights
    if verbose:
        print('Pruning Summary', flush=True)
        print(tabulate(prune_stat, headers='keys'), flush=True)
        print(
            f'Percent Pruned Globally: {global_prune_percent:.2f}', flush=True)


def fprune_fixed_amount_res(model, amount, verbose=True, glob=True):
    parameters_to_prune, num_global_weights, layers_w_count = get_prune_params_res(
        model)

    if glob:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=math.floor(amount * num_global_weights))
    else:
        for i, (m, n) in enumerate(parameters_to_prune):
            prune.l1_unstructured(m, name=n, amount=math.floor(
                amount * layers_w_count[i][1]))
    #masks = dict(model.named_buffers())['weight_mask']
    # for i, (named, param) in enumerate(model.named_parameters()):
    #    #for name, param in namedp:
    #    param.data.copy_(param.data*masks[i])

    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}

    # Pruning is done in-place, thus parameters_to_prune is updated
    for layer, weight_name in parameters_to_prune:

        num_layer_zeros = torch.sum(getattr(layer, weight_name) == 0.0).item()
        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(layer.__str__())
        prune_stat['Weight Name'].append(weight_name)
        prune_stat['Percent Pruned'].append(
            f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

    global_prune_percent = num_global_zeros / num_global_weights
    if verbose:
        print('Pruning Summary', flush=True)
        print(tabulate(prune_stat, headers='keys'), flush=True)
        print(
            f'Percent Pruned Globally: {global_prune_percent:.2f}', flush=True)


def fprune_fixed_amount(model, amount, verbose=True, glob=True):
    parameters_to_prune, num_global_weights, layers_w_count = get_prune_params(
        model)

    if glob:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=math.floor(amount * num_global_weights))
    else:
        for i, (m, n) in enumerate(parameters_to_prune):
            prune.l1_unstructured(m, name=n, amount=math.floor(
                amount * layers_w_count[i][1]))
    #masks = dict(model.named_buffers())['weight_mask']
    # for i, (named, param) in enumerate(model.named_parameters()):
    #    #for name, param in namedp:
    #    param.data.copy_(param.data*masks[i])

    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}

    # Pruning is done in-place, thus parameters_to_prune is updated
    for layer, weight_name in parameters_to_prune:

        num_layer_zeros = torch.sum(getattr(layer, weight_name) == 0.0).item()
        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(layer.__str__())
        prune_stat['Weight Name'].append(weight_name)
        prune_stat['Percent Pruned'].append(
            f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

    global_prune_percent = num_global_zeros / num_global_weights
    if verbose:
        print('Pruning Summary', flush=True)
        print(tabulate(prune_stat, headers='keys'), flush=True)
        print(
            f'Percent Pruned Globally: {global_prune_percent:.2f}', flush=True)


"""
Hadamard Mult of Mask and Attributes,
then return zeros
"""


def get_prune_summary_res(model):
    num_global_zeros = 0
    parameters_to_prune, num_global_weights, _ = get_prune_params_res(model)

    masks = dict(model.named_buffers())

    for i, (layer, weight_name) in enumerate(parameters_to_prune):
        attr = getattr(layer, weight_name)
        try:
            attr *= masks[list(masks)[i]]
        except Exception as e:
            print(e)

        num_global_zeros += torch.sum(attr == 0.0).item()

    return num_global_zeros, num_global_weights


def get_prune_summary(model):
    num_global_zeros = 0
    parameters_to_prune, num_global_weights, _ = get_prune_params(model)

    masks = dict(model.named_buffers())

    for i, (layer, weight_name) in enumerate(parameters_to_prune):
        attr = getattr(layer, weight_name)
        try:
            attr *= masks[list(masks)[i]]
        except Exception as e:
            print(e)

        num_global_zeros += torch.sum(attr == 0.0).item()

    return num_global_zeros, num_global_weights


def get_prune_params_res(model):
    layers = []
    layers_weight_count = []

    num_global_weights = 0

    modules = list(model.modules())

    for layer in modules:

        is_sequential = type(layer) == nn.Sequential

        is_itself = type(layer) == type(model) if len(modules) > 1 else False

        is_conv = isinstance(layer, nn.Conv2d)
        is_Lin = isinstance(layer, nn.Linear)

        if (not is_sequential) and (not is_itself) and (is_conv or is_Lin):
            for name, param in layer.named_parameters():

                field_name = name.split('.')[-1]

                # This might break if someone does not adhere to the naming
                # convention where weights of a module is stored in a field
                # that has the word 'weight' in it

                if 'weight' in field_name and param.requires_grad:

                    if field_name.endswith('_orig'):
                        field_name = field_name[:-5]

                    # Might remove the param.requires_grad condition in the future

                    layers.append((layer, field_name))

                    layer_weight_count = torch.numel(param)

                    layers_weight_count.append((layer, layer_weight_count))

                    num_global_weights += layer_weight_count

    return layers, num_global_weights, layers_weight_count


def get_prune_params(model):
    layers = []
    layers_weight_count = []

    num_global_weights = 0

    modules = list(model.modules())

    for layer in modules:

        is_sequential = type(layer) == nn.Sequential

        is_itself = type(layer) == type(model) if len(modules) > 1 else False

        if (not is_sequential) and (not is_itself):
            for name, param in layer.named_parameters():

                field_name = name.split('.')[-1]

                # This might break if someone does not adhere to the naming
                # convention where weights of a module is stored in a field
                # that has the word 'weight' in it

                if 'weight' in field_name and param.requires_grad:

                    if field_name.endswith('_orig'):
                        field_name = field_name[:-5]

                    # Might remove the param.requires_grad condition in the future

                    layers.append((layer, field_name))

                    layer_weight_count = torch.numel(param)

                    layers_weight_count.append((layer, layer_weight_count))

                    num_global_weights += layer_weight_count

    return layers, num_global_weights, layers_weight_count


def calculate_metrics(score, ytrue, yraw, ypred):
    if 'Loss' in score:
        loss = nn.CrossEntropyLoss()
        score['Loss'].append(loss(yraw, ytrue))
    if 'Accuracy' in score:
        score['Accuracy'].append(skmetrics.accuracy_score(ytrue, ypred))
    if 'Balanced Accuracy' in score:
        score['Balanced Accuracy'].append(
            skmetrics.balanced_accuracy_score(ytrue, ypred))
    if 'Precision Micro' in score:
        score['Precision Micro'].append(skmetrics.precision_score(ytrue,
                                                                  ypred,
                                                                  average='micro',
                                                                  zero_division=0))
    if 'Recall Micro' in score:
        score['Recall Micro'].append(skmetrics.recall_score(ytrue,
                                                            ypred,
                                                            average='micro',
                                                            zero_division=0))
    if 'Precision Macro' in score:
        score['Precision Macro'].append(skmetrics.precision_score(ytrue,
                                                                  ypred,
                                                                  average='macro',
                                                                  zero_division=0))
    if 'Recall Macro' in score:
        score['Recall Macro'].append(skmetrics.recall_score(ytrue,
                                                            ypred,
                                                            average='macro',
                                                            zero_division=0))

    return score


def log_obj(path, obj):
    # pass
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #
    with open(path, 'wb') as file:
        if isinstance(obj, nn.Module):
            torch.save(obj, file)
        else:
            pickle.dump(obj, file)
