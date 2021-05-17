import numpy as np
import torch
torch.manual_seed(111)
np.random.seed(111)
from util import average_weights, average_weights_masks, create_model, copy_model, log_obj, evaluate, fevaluate, fed_avg, lottery_fl_v2, lottery_fl_v3, lottery_fl_v3_res


class Server():

    def __init__(self, args, clients, test_loader=None):

        self.args = args
        self.comm_rounds = args.comm_rounds
        self.num_clients = args.num_clients
        self.frac = args.frac
        self.clients = clients
        self.client_data_num = []
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros(args.comm_rounds)
        self.client_accuracies = np.zeros((self.args.num_clients, self.args.comm_rounds))
        self.selected_client_tally = np.zeros((self.args.comm_rounds, self.args.num_clients))
        self.good_client_tally = np.zeros((self.args.num_clients, self.args.comm_rounds))  ### Determines good rounds for clients
        self.test_loader = test_loader

        for client in self.clients:
            self.client_data_num.append(len(client.train_loader))
        self.client_data_num = np.array(self.client_data_num)

        # The extra 1 entry in client_models and global_models are used to store
        # the results after last communication round
        self.client_models = np.zeros((self.comm_rounds + 1, self.num_clients), dtype='object')
        self.global_models = None  # np.zeros((self.comm_rounds + 1,), dtype='object')
        self.global_model_mem = np.zeros((self.comm_rounds + 1,), dtype='object')   ### Model memories

        self.init_model = create_model(args.dataset, args.arch)

        self.global_models = self.init_model
        self.global_init_model = copy_model(self.init_model, args.dataset, args.arch)
        #self.global_model_mem[0] = self.init_model   ### First round is init
        ###print(clients, self.num_clients)
        for i in range(len(self.global_model_mem)):
            self.global_model_mem[i] = copy_model(self.init_model, args.dataset, args.arch)
        assert self.num_clients == len(clients), "Number of client objects does not match command line input"

    def server_update(self):
        self.elapsed_comm_rounds += 1
        # Recording the update and storing them in record
        self.global_models.train()
        for i in range(0, self.comm_rounds):
            update_or_not = [0] * self.num_clients
            # Randomly select a fraction of users to update
            num_selected_clients = max(int(self.frac * self.num_clients), 1)
            idx_list = np.random.choice(range(self.num_clients),
                                        num_selected_clients,
                                        replace=False)
            print("type of the idx: " + str(type(idx_list)), flush=True) ###
            for idx in idx_list:
                update_or_not[idx] = 1

            print('-------------------------------------', flush=True)
            print(f'Communication Round #{i}', flush=True)
            print('-------------------------------------', flush=True)
            for j in range(len(update_or_not)):

                if update_or_not[j]:
                    if self.args.avg_logic == "standalone":
                        self.clients[j].client_update(self.clients[j].model, self.global_init_model, i)
                    else:
                        self.clients[j].client_update(self.global_models, self.global_init_model, i)
                else:
                    pass
                    # copy_model(self.clients[j].model, self.args.dataset, self.args.arch)

            models = []
            self.selected_client_tally[i, idx_list] += 1
            ### print(self.clients[idx_list])
            ###for m in self.clients[idx_list]:
            for idx in idx_list:
                m = self.clients[idx]
                models.append(m.model)
            if self.args.avg_logic == "fed_avg":
                self.global_models = fed_avg(list(map(lambda x: x.model, self.clients)), self.args.dataset,
                                             self.args.arch,
                                             self.client_data_num)
            elif self.args.avg_logic == 'lottery_fl_v2':
                self.global_models = lottery_fl_v2(self.global_models, models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            elif self.args.avg_logic == 'lottery_fl_v3':
                self.global_models = lottery_fl_v3(self.global_models, models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            elif self.args.avg_logic == "standalone":
                pass #no averaging in the server
            else:
                self.global_models = average_weights(models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            
            del models

            eval_score = evaluate(self.global_models,
                                  self.test_loader,
                                  verbose=self.args.test_verbosity)
            print(f"Server accuracies over the batch + avg at the end: {eval_score['Accuracy']}")
            self.accuracies[i] = eval_score['Accuracy'][-1]

            for k, m in enumerate(self.clients):
                if k in idx_list:
                    self.client_accuracies[k][i] = m.evaluate()
                else:
                    if i == 0:
                        pass
                    else:
                        self.client_accuracies[k][i] = self.client_accuracies[k][i - 1]
            print(f"Mean client accs: {self.client_accuracies.mean(axis=0)[i]}")


    def fserver_update(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.elapsed_comm_rounds += 1
        # Recording the update and storing them in record
        self.global_models.train()
        for i in range(0, self.comm_rounds):
            self.global_model_mem[i] = copy_model(self.global_models, self.args.dataset, self.args.arch).to(device)#self.global_models.to(device) ### memory of models
            update_or_not = [0] * self.num_clients
            # Randomly select a fraction of users to update
            num_selected_clients = max(int(self.frac * self.num_clients), 1)
            idx_list = np.random.choice(range(self.num_clients),
                                        num_selected_clients,
                                        replace=False)
            print("type of the idx: " + str(type(idx_list)), flush=True) ###
            for idx in idx_list:
                update_or_not[idx] = 1

            print('FF---------------------------------FF', flush=True)
            print(f'Communication Round #{i}', flush=True)
            print('-------------------------------------', flush=True)
            for j in range(len(update_or_not)):

                if update_or_not[j]:
                    if self.args.avg_logic == "standalone":
                        self.clients[j].fclient_update(self.clients[j].model, self.global_init_model, i)
                    elif self.args.avg_logic == "lottery_fl_v3_res":
                        self.clients[j].fclient_update_res(self.global_models, self.global_init_model, i)
                    else:
                        self.clients[j].fclient_update(self.global_models, self.global_init_model, i)
                        #print(self.global_model_mem, flush=True)
                        #self.clients[j].ffclient_update(self.global_model_mem, self.global_init_model, i, self.good_client_tally[j])
                        if i == 0 and self.clients[j].prune_rates[i] != 0.0:
                            self.good_client_tally[j][i] = 1
                        elif self.clients[j].prune_rates[i] != self.clients[j].prune_rates[i-1]:
                            self.good_client_tally[j][i] = 1
                else:
                    pass
                    # copy_model(self.clients[j].model, self.args.dataset, self.args.arch)

            models = []
            self.selected_client_tally[i, idx_list] += 1
            ### print(self.clients[idx_list])
            ###for m in self.clients[idx_list]:
            for idx in idx_list:
                m = self.clients[idx]
                models.append(m.model)
            if self.args.avg_logic == "fed_avg":
                self.global_models = fed_avg(list(map(lambda x: x.model, self.clients)), self.args.dataset,
                                             self.args.arch,
                                             self.client_data_num)
            elif self.args.avg_logic == 'lottery_fl_v2':
                self.global_models = lottery_fl_v2(self.global_models, models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            elif self.args.avg_logic == 'lottery_fl_v3':
                self.global_models = lottery_fl_v3(self.global_models, models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            elif self.args.avg_logic == 'lottery_fl_v3_res':
                self.global_models = lottery_fl_v3_res(self.global_models, models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            elif self.args.avg_logic == "standalone":
                pass #no averaging in the server
            else:
                self.global_models = average_weights(models, self.args.dataset,
                                                          self.args.arch,
                                                          self.client_data_num[idx_list])
            del models

            eval_score = fevaluate(self.global_models,
                                  self.test_loader,
                                  verbose=self.args.test_verbosity)
            print(f"Server accuracies over the batch + avg at the end: {eval_score['Accuracy']}")
            self.accuracies[i] = eval_score['Accuracy'][-1]

            for k, m in enumerate(self.clients):
                if k in idx_list:
                    self.client_accuracies[k][i] = m.fevaluate()
                    print("client #: ", k, "with ", m.prune_rates[-1])
                else:
                    if i == 0:
                        pass
                    else:
                        self.client_accuracies[k][i] = self.client_accuracies[k][i - 1]
            print(f"Mean client accs: {self.client_accuracies.mean(axis=0)[i]}")
        self.global_model_mem[i] = copy_model(self.global_models, self.args.dataset, self.args.arch).to(device) ### add last model

            
    def test_server_update(self):
                
        self.elapsed_comm_rounds += 1
        # Recording the update and storing them in record
        for i in range(1, self.comm_rounds+1):
            update_or_not = [0] * self.num_clients
            # Randomly select a fraction of users to update
            num_selected_clients = max(int(self.frac * self.num_clients), 1)
            idx_list = np.random.choice(range(self.num_clients), num_selected_clients, replace=False)
            for idx in idx_list:
                update_or_not[idx] = 1
           
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', flush=True)
            print(f'Communication Round #{i}', flush=True)
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', flush=True)
            for j in range(len(update_or_not)):
                
                if update_or_not[j]:
                    #number_zeros, number_gl_weights = get_prune_summary(self.global_models[i-1])
                    #print(f'num_zeros:: {number_zeros}    num_weights:: {number_gl_weights}', flush=True)
                    self.client_models[i][j] = self.clients[j].test_client_update(self.global_models, self.global_init_model, i)
                else:
                    self.client_models[i][j] = copy_model(self.clients[j].model, self.args.dataset, self.args.arch)

            models = self.client_models[i][idx_list]
            self.global_models = average_weights_masks(models, 
                                                    self.args.dataset, 
                                                    self.args.arch,
                                                    self.client_data_num)

            eval_score = fevaluate(self.global_models,
                                  self.test_loader,
                                  verbose=self.args.test_verbosity)
            self.accuracies[i-1] = eval_score['Accuracy'][-1]
            client_model_path = './log/server/client_models/client_models.model_list'
            server_model_path = f'./log/server/server_models/average_model_round{self.elapsed_comm_rounds}.model_list'
            log_obj(client_model_path, self.client_models)
            log_obj(server_model_path, self.global_models)
            
            for k, m in enumerate(self.clients):
                if k in idx_list:
                    self.client_accuracies[k][i-1] = m.fevaluate()
                else:
                    if i == 0:
                        pass
                    else:
                        self.client_accuracies[k][i-1] = self.client_accuracies[k][i - 2]
            print(f"Mean client accs: {self.client_accuracies.mean(axis=0)[i-1]}")
            