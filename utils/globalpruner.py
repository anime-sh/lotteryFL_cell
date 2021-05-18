import torch
from torch import nn
from torch.nn import Module
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from utils import get_prune_params, create_model, copy_model, prune_fixed_amount
from tabulate import tabulate


class GlobalPruner(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold, orig_weights):
        super().__init__()
        self.threshold = threshold
        self.original_signs = self.get_signs_from_tensor(orig_weights)

    def get_signs_from_tensor(self, t: torch.Tensor):
        return torch.gt(t, -0.0000000).view(-1)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        large_weight_mask = torch.gt(torch.abs(t), self.threshold).view(-1)
        large_mask_signs = self.get_signs_from_tensor(t)
        mask.view(-1)[:] = mask*((~(large_mask_signs ^
                                    self.original_signs))*(large_weight_mask))

        return mask


def globalPrunerStructured(module, name='weight', threshold=0.1, orig_weights=None):
    """
        Taken from https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        Takes: current module (module), name of the parameter to prune (name)

    """
    GlobalPruner.apply(module, name, threshold, orig_weights)
    return module


def get_parameters(module, name="weight_orig") -> nn.Parameter:
    for n, params in module.named_parameters():
        if n == name:
            return params


def super_prune(model, init_model, name="weight", threshold=0.2,verbose = True):

    params, num_global_weights, layers_w_count = get_prune_params(model)
    init_params, _, _ = get_prune_params(init_model)

    for idx, (param, name) in enumerate(params):
        orig_params = get_parameters(init_params[idx][0], "weight_orig")
        orig_params = orig_params[param.weight_mask.to(torch.bool)]
        globalPrunerStructured(param, "weight", threshold, orig_params)
        

    if verbose:
        num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
        global_prune_percent, layer_prune_percent = 0, 0
        prune_stat = {'Layers': [],
                    'Weight Name': [],
                    'Percent Pruned': [],
                    'Total Pruned': []}

        for layer, weight_name in params:

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
        
        print('Pruning Summary', flush=True)
        print(tabulate(prune_stat, headers='keys'), flush=True)
        print(
            f'Percent Pruned Globally: {global_prune_percent:.2f}', flush=True)


