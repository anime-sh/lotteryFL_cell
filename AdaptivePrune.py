import torch
from timeit import default_timer as timer


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
    print(f"Total: {total_param_in_use}/{total_all_param} = {total_param_in_use / total_all_param}")

    return total_param_in_use / total_all_param


def AdaptivePrune(self, list_est_time, list_loss, list_acc, list_model_size):
    svdata, pvdata = self.ip_train_loader.len_data, self.config.IP_DATA_BATCH * \
        self.config.CLIENT_BATCH_SIZE
    assert svdata >= pvdata, "server data ({}) < required data ({})".format(
        svdata, pvdata)
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
