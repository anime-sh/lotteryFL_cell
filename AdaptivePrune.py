import torch
from timeit import default_timer as timer
from typing import Tuple

def calc_num_all_active_params(self, count_bias):
    total_param = 0
    for layer in self.param_layers:
        num_bias = layer.bias.nelement() if layer.bias is not None and count_bias else 0
        num_weight = layer.num_weight if hasattr(
            layer, "num_weight") else layer.weight.nelement()
        num_params = num_weight + num_bias
        total_param += num_params

    return total_param


def disp_num_params(model):
    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefx in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param
        print(f"{layer_prefx} remaining: {layer_param_in_use}/{layer_all_param} = {(layer_param_in_use / layer_all_param)} ")
    print(
        f"Total: {total_param_in_use}/{total_all_param} = {total_param_in_use / total_all_param}")

    return total_param_in_use / total_all_param


# might not be relevant
def init_AdaptivePrune(self, list_est_time, list_loss, list_acc, list_model_size):
    svdata, pvdata = self.ip_train_loader.len_data, self.config.IP_DATA_BATCH * \
        self.config.CLIENT_BATCH_SIZE
    assert svdata >= pvdata, f"server data ({svdata}) < required data ({pvdata})"
    server_inputs, server_outputs = [], []
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for _ in range(self.config.IP_DATA_BATCH):
        inp, out = self.ip_train_loader.get_next_batch()
        server_inputs.append(inp.to(dev))
        server_outputs.append(out.to(dev))

    prev_density = None
    prev_num = 5
    prev_ind = []
    start = timer()
    ip_start_adj_round = None

    for server_i in range(1, self.config.IP_MAX_ROUNDS + 1):
        model_size = self.model.calc_num_all_active_params(True)
        list_est_time.append(0)
        list_model_size.append(model_size)

        if (server_i - 1) % self.config.EVAL_DISP_INTERVAL == 0:
            # test data not observable to clients, this evaluation does not happen in real systems
            loss, acc = self.model.evaluate(self.ip_test_loader)
            train_loss, train_acc = self.model.evaluate(
                zip(server_inputs, server_outputs))
            list_loss.append(loss)
            list_acc.append(acc)
            if ip_start_adj_round is None and train_acc >= self.config.ADJ_THR_ACC:
                ip_start_adj_round = server_i
                print("Start reconfiguration in initial pruning at round {}.".format(
                    server_i - 1))
            print("Initial pruning round {}. Accuracy = {}. Loss = {}. Train accuracy = {}. Train loss = {}. "
                  "Elapsed time = {}.".format(server_i - 1, acc, loss, train_acc, train_loss, timer() - start))

        for server_inp, server_out in zip(server_inputs, server_outputs):
            list_grad = self.ip_optimizer_wrapper.step(server_inp, server_out)
            for (key, param), g in zip(self.model.named_parameters(), list_grad):
                assert param.size() == g.size()
                self.ip_control.accumulate(key, g ** 2)

        if ip_start_adj_round is not None and (server_i - ip_start_adj_round) % self.config.IP_ADJ_INTERVAL == 0:
            self.ip_control.adjust(self.config.MAX_DEC_DIFF)
            cur_density = disp_num_params(self.model)

            if prev_density is not None:
                prev_ind.append(
                    abs(cur_density / prev_density - 1) <= self.config.IP_THR)
            prev_density = cur_density

            if len(prev_ind) >= prev_num and all(prev_ind[-prev_num:]):
                print("Early-stopping initial pruning at round {}.".format(server_i - 1))
                del list_loss[-1]
                del list_acc[-1]
                break

    len_pre_rounds = len(list_acc)
    print("End initial pruning. Total rounds = {}. Total elapsed time = {}.".format(
        len_pre_rounds * self.config.EVAL_DISP_INTERVAL, timer() - start))


# further prune steps
def FurtherPrune(self, idx, list_sd, list_num_proc, lr, list_accumulated_sgrad, start, list_loss, list_acc, list_est_time, list_model_size, is_adj_round, density_limit=None):
	alg_start = timer()
	for d in list_accumulated_sgrad:
		for k, sg in d.items():
			self.control.accumulate(k, sg)
	print("Running adaptive pruning algorithm")
	max_dec_diff = self.config.MAX_DEC_DIFF * (0.5 ** (idx / self.config.ADJ_HALF_LIFE))
	self.control.adjust(max_dec_diff, max_density=density_limit)
	print(f"Total alg time = {timer() - alg_start}. Max density = {density_limit}.")
	print(f"Num params:{disp_num_params(self.model)}")

#############heap################3
import heapq
from typing import Iterable


class HeapQueue:
    def __init__(self, init_h: Iterable):
        self.h = [(-val, index) for index, val in init_h]
        heapq.heapify(self.h)

    def replace_largest(self, new_val):
        heapq.heapreplace(self.h, (-new_val, self.max_index))

    def pop(self):
        heapq.heappop(self.h)

    @property
    def max_index(self):
        return self.h[0][1]

    @property
    def max_val(self):
        return -self.h[0][0]

    def __repr__(self):
        return f"HeapQueue instance containing data {self.h}."

###########control#####################


def process_layer(layer, layer_prefix, sgrad: dict, coeff, dtp) -> Tuple[float, float, list, torch.Tensor]:
    w_name = "{}.weight".format(layer_prefix)
    b_name = "{}.bias".format(layer_prefix)
    sqg = sgrad[w_name]
    iu_mask, niu_mask = layer.mask == 1., layer.mask == 0.
    num_iu, num_niu = iu_mask.sum().item(), niu_mask.sum().item()

    # Decrease
    max_dec_num = int(dtp * num_iu)
    w_iu = layer.weight[iu_mask]  # use magnitude
    w_thr = torch.sort(torch.abs(w_iu))[0][max_dec_num]
    tbk_mask = (torch.abs(layer.weight) >= w_thr) * iu_mask
    tba_dec_mask = (torch.abs(layer.weight) < w_thr) * iu_mask

    # Increase
    tba_inc_mask = niu_mask

    total_sqg = sqg[tbk_mask].sum().item()
    if b_name in sgrad.keys():
        total_sqg += sgrad[b_name].sum().item()
    total_time = coeff * tbk_mask.sum().item()
    tba_mask = tba_dec_mask + tba_inc_mask
    tba_values, tba_indices = sqg[tba_mask], tba_mask.nonzero()
    sorted_tba_values, sort_perm = torch.sort(tba_values, descending=True)
    sorted_tba_indices = tba_indices[sort_perm]

    layer.prune_by_pct(dtp)

    return total_sqg, total_time, sorted_tba_values.tolist(), sorted_tba_indices


def architecture_search(model, sum_sqg, sum_time, list_coefficient, list_tba_values, list_tba_indices,
                        max_density=None):
    list_len = [len(tba) for tba in list_tba_values]
    list_iter = [iter(tba) for tba in list_tba_values]
    # number of params to be added/removed
    list_n = [0] * len(list_len)

    heap = HeapQueue([(index, next(_iter) / _coeff) for index, (_iter, _coeff, _len) in
                      enumerate(zip(list_iter, list_coefficient, list_len)) if _len > 0])
    numerator = sum_sqg
    denominator = sum_time

    num_params, max_num = None, None
    if max_density is not None:
        num_params, max_num = model.calc_num_prunable_params(False)
        max_num = int(max_num * max_density)

    end_condition = False
    while not end_condition:
        obj_val = numerator / denominator
        pos, val = heap.max_index, heap.max_val
        if val > obj_val:
            if max_num is not None:
                if num_params > max_num:
                    print("Exceeds max num")
                    break
                else:
                    num_params += 1
            coeff = list_coefficient[pos]
            numerator += val * coeff
            denominator += coeff
            list_n[pos] += 1
            if list_n[pos] == list_len[pos]:
                heap.pop()
            else:
                heap.replace_largest(next(list_iter[pos]) / coeff)
        else:
            end_condition = True

    print(list_n)

    for layer, tba_indices, tba_n in zip(model.prunable_layers, list_tba_indices, list_n):
        layer.mask[tba_indices[:tba_n].t().tolist()] = 1.


def main_control(model, squared_grad_dict: dict, config, dec_thr_pct, max_density=None):
    sum_sqg = 0
    sum_time = config.TIME_CONSTANT
    list_tba_values, list_tba_indices = [], []
    list_coefficient = []

    proc_start = timer()
    comp_coeff_iter = iter(config.COMP_COEFFICIENTS)
    comm_coeff = config.COMM_COEFFICIENT
    for layer, layer_prefix in zip(model.param_layers, model.param_layer_prefixes):
        if layer_prefix in model.prunable_layer_prefixes:
            coeff = comm_coeff + next(comp_coeff_iter)
            sqg, time, sorted_tba_values, sorted_tba_indices = process_layer(layer, layer_prefix, squared_grad_dict,
                                                                             coeff, dec_thr_pct)
            sum_sqg += sqg
            sum_time += time

            list_coefficient.append(coeff)
            list_tba_values.append(sorted_tba_values)
            list_tba_indices.append(sorted_tba_indices)
        else:
            w_name = "{}.weight".format(layer_prefix)
            b_name = "{}.bias".format(layer_prefix)
            sqg = squared_grad_dict[w_name]
            sum_sqg += sqg.sum().item()
            if b_name in squared_grad_dict.keys():
                sum_sqg += squared_grad_dict[b_name].sum().item()

    print("\tProcessing layers, time = {}.".format(timer() - proc_start))
    nas_start = timer()
    architecture_search(model, sum_sqg, sum_time, list_coefficient, list_tba_values, list_tba_indices, max_density)
    print("\tNAS time = {}.".format(timer() - nas_start))

class ControlModule:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.squared_grad_dict = dict()

    @torch.no_grad()
    def accumulate(self, key, sgrad):
        if key in self.squared_grad_dict.keys():
            self.squared_grad_dict[key] += sgrad
        else:
            self.squared_grad_dict[key] = sgrad

    def adjust(self, dec_thr_pct, max_density=None):
        main_control(self.model, self.squared_grad_dict, self.config, dec_thr_pct, max_density)
        self.squared_grad_dict = dict()
