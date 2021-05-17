import math
from util import train, ftrain, evaluate, fevaluate, fprune_fixed_amount, fprune_fixed_amount_res, prune_fixed_amount, copy_model, \
                 create_model, get_prune_summary, log_obj, merge_models, get_prune_summary_res
import numpy as np
import torch.nn.utils.prune as prune
import torch 
import torch.nn as nn
import copy

torch.manual_seed(111)
np.random.seed(111)

class Client:
    def __init__(self, 
                 args, 
                 train_loader, 
                 test_loader,
                 client_id=None):
        self.args = args#copy.copy(args) #args
        print("Creating model for client "+ str(client_id))
        self.model = create_model(self.args.dataset, self.args.arch)
        print("Copying model for client "+ str(client_id))
        self.init_model = copy_model(self.model, self.args.dataset, self.args.arch)
        print("Done Copying model for client "+ str(client_id))
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.client_id = client_id
        self.elapsed_comm_rounds = 0
        self.accuracies = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.losses = np.zeros((args.comm_rounds, self.args.client_epoch))
        self.prune_rates = np.zeros(args.comm_rounds)
        assert self.model, "Something went wrong and the model cannot be initialized"

        # This is a sanity check that we're getting proper data. Once we are confident about this, we can delete this.
        # train_classes =  self.get_class_counts('train')
        # test_classes  =  self.get_class_counts('test')
        # assert len(train_classes.keys()) == 2,\
        #     f'Client {self.client_id} should have 2 classes in train set but has {len(train_classes.keys())}.'
        # assert len(test_classes.keys()) == 2,\
        #     f'Client {self.client_id} should have 2 classes in test set but has {len(test.keys())}.'
        # assert set(train_classes.keys()) == set(test_classes.keys()),\
        #     f'Client {self.client_id} has different keys for train ({train_classes.keys()}) and test ({test_classes.keys()}).'


    def client_update(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'***** Client #{self.client_id} *****', flush=True)
        self.model = copy_model(global_model,
                                self.args.dataset,
                                self.args.arch,
                                dict(self.model.named_buffers()))
        
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        #prune_step = math.floor(num_params * self.args.prune_step)
        ###need to prune
        eval_score = evaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        
        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
            prune_fraction = min(self.args.prune_step, 0.001 + self.args.prune_percent - cur_prune_rate)
            prune_fixed_amount(self.model, 
                               prune_fraction,
                               verbose=self.args.prune_verbosity, glob=True)
            self.model = copy_model(global_init_model,
                                    self.args.dataset,
                                    self.args.arch,
                                    dict(self.model.named_buffers()))
            ###need to prune
        losses = []
        accuracies = []
        for i in range(self.args.client_epoch):
            train_score = train(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
           
            
        mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")


        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        self.prune_rates[round_index:] = cur_prune_rate


        return copy_model(self.model, self.args.dataset, self.args.arch)

    def fclient_update(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'FF***** Client #{self.client_id} *****FF', flush=True)
        self.model = copy_model(global_model,
                                self.args.dataset,
                                self.args.arch,
                                dict(self.model.named_buffers()))
        
        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        print(cur_prune_rate, flush=True)
        #prune_step = math.floor(num_params * self.args.prune_step)
        eval_score = fevaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        #prune_fraction = 0
        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
            prune_fraction = min(self.args.prune_step, self.args.prune_percent - cur_prune_rate)
            cur_prune_rate = cur_prune_rate + prune_fraction
            fprune_fixed_amount(self.model, 
                               prune_fraction,
                               verbose=True, glob=True)#self.args.prune_verbosity, glob=True)
            
            self.model = copy_model(global_init_model,
                                    self.args.dataset,
                                    self.args.arch,
                                    dict(self.model.named_buffers()))
            
        losses = []
        accuracies = []
        for i in range(self.args.client_epoch):
            train_score = ftrain(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            
            fprune_fixed_amount(self.model, 
                               cur_prune_rate,#prune_fraction,
                               verbose=self.args.prune_verbosity, glob=True)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
           
        print(cur_prune_rate, flush=True)
        mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")


        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        self.prune_rates[round_index:] = cur_prune_rate


        return copy_model(self.model, self.args.dataset, self.args.arch)
    
    def fclient_update_res(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'FF***** Client #{self.client_id} *****FF', flush=True)
        self.model = copy_model(global_model,
                                self.args.dataset,
                                self.args.arch,
                                dict(self.model.named_buffers()))
        
        num_pruned, num_params = get_prune_summary_res(self.model)
        cur_prune_rate = num_pruned / num_params
        print(cur_prune_rate, flush=True)
        #prune_step = math.floor(num_params * self.args.prune_step)
        eval_score = fevaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        #prune_fraction = 0
        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
            prune_fraction = min(self.args.prune_step, self.args.prune_percent - cur_prune_rate)
            cur_prune_rate = cur_prune_rate + prune_fraction
            fprune_fixed_amount_res(self.model, 
                               prune_fraction,
                               verbose=True, glob=True)#self.args.prune_verbosity, glob=True)
            
            self.model = copy_model(global_init_model,
                                    self.args.dataset,
                                    self.args.arch,
                                    dict(self.model.named_buffers()))
            
        losses = []
        accuracies = []
        for i in range(self.args.client_epoch):
            train_score = ftrain(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            
            fprune_fixed_amount_res(self.model, 
                               cur_prune_rate,#prune_fraction,
                               verbose=self.args.prune_verbosity, glob=True)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
           
        print(cur_prune_rate, flush=True)
        mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)

        num_pruned, num_params = get_prune_summary_res(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")


        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        self.prune_rates[round_index:] = cur_prune_rate


        return copy_model(self.model, self.args.dataset, self.args.arch)
    
    def ffclient_update(self, global_models, global_init_model, round_index, tallys):
        self.elapsed_comm_rounds += 1
        global_model = merge_models(global_models, self.args.dataset, self.args.arch, tallys, round_index)
        
        num_pruned, num_params = get_prune_summary(self.model)
        
        weights = dict(global_model.named_parameters())

        print(f'FF***** Client #{self.client_id} *****FF', flush=True)
        self.model = create_model(self.args.dataset, self.args.arch)
        for name, param in self.model.named_parameters():
            param.data.copy_(weights[name].copy())
        
                        #copy_model(global_model,
                         #       self.args.dataset,
                          #      self.args.arch)#,
                                #dict(self.model.named_buffers()))
        
        cur_prune_rate = num_pruned / num_params
        print(cur_prune_rate, flush=True)
        #prune_step = math.floor(num_params * self.args.prune_step)
        eval_score = fevaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        #prune_fraction = 0
        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
            prune_fraction = min(self.args.prune_step, self.args.prune_percent - cur_prune_rate)
            cur_prune_rate = cur_prune_rate + prune_fraction
            fprune_fixed_amount(self.model, 
                               cur_prune_rate,#prune_fraction,
                               verbose=True, glob=True)#self.args.prune_verbosity, glob=True)
            
            self.model = copy_model(global_init_model,
                                    self.args.dataset,
                                    self.args.arch,
                                    dict(self.model.named_buffers()))
            
        losses = []
        accuracies = []
        for i in range(self.args.client_epoch):
            train_score = ftrain(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            
            fprune_fixed_amount(self.model, 
                               cur_prune_rate,#prune_fraction,
                               verbose=self.args.prune_verbosity, glob=True)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
           
        print(cur_prune_rate, flush=True)
        mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")


        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        self.prune_rates[round_index:] = cur_prune_rate


        return copy_model(self.model, self.args.dataset, self.args.arch)
    
    
    def sclient_update(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'FF***** Client #{self.client_id} *****FF', flush=True)
        #self.model = copy_model(global_model,
        #                        self.args.dataset,
        #                        self.args.arch,
        #                        dict(self.model.named_buffers()))
        
        #num_pruned, num_params = get_prune_summary(self.model)
        #cur_prune_rate = num_pruned / num_params
        #prune_step = math.floor(num_params * self.args.prune_step)
        #eval_score = evaluate(self.model, 
        #                 self.test_loader,
        #                 verbose=self.args.test_verbosity)
        
        #if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
        #    # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
        #    prune_fraction = min(self.args.prune_step, 0.001 + self.args.prune_percent - cur_prune_rate)
        #    fprune_fixed_amount(self.model, 
        #                       prune_fraction,
        #                       verbose=self.args.prune_verbosity, glob=True)
            
         #   self.model = copy_model(global_init_model,
         #                           self.args.dataset,
         #                           self.args.arch,
         #                           dict(self.model.named_buffers()))
        prune_fraction = 0
        losses = []
        accuracies = []
        for i in range(self.args.client_epoch):
            train_score = ftrain(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            fprune_fixed_amount(self.model, 
                               prune_fraction,
                               verbose=self.args.prune_verbosity, glob=True)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
           
            
        #mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        #client_mask = dict(self.model.named_buffers())
        #log_obj(mask_log_path, client_mask)

        #num_pruned, num_params = get_prune_summary(self.model)
        #cur_prune_rate = num_pruned / num_params
        #prune_step = math.floor(num_params * self.args.prune_step)
        #print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")


        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        #self.prune_rates[round_index:] = cur_prune_rate


        return copy_model(self.model, self.args.dataset, self.args.arch)

    def test_client_update(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'$$$$$ Client #{self.client_id} $$$$$', flush=True)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        prune_fraction = cur_prune_rate
        self.model = copy_model(global_model,
                                self.args.dataset,
                                #self.args.arch, dict(self.model.named_buffers()))
                                self.args.arch, dict(global_model.named_buffers()))
        for mod in list(self.model.modules()):
            if type(mod) != nn.Sequential and type(mod) != type(self.model):
                for name, param in mod.named_parameters():
                    if 'weight' in name: 
                        prune.remove(mod, 'weight')
        eval_flag = 0
        eval_score = fevaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        fprune_fixed_amount(self.model,
                               0,#prune_step,
                               verbose=self.args.prune_verbosity)

        eval_log_path = f'./log/clients/client{self.client_id}/'\
                        f'round{self.elapsed_comm_rounds}/'\
                        f'eval_score_round{self.elapsed_comm_rounds}.pickle'
        log_obj(eval_log_path, eval_score)


        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            self.args.acc_thresh = 0.5
            # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
            prune_fraction = cur_prune_rate + min(self.args.prune_step, 0.001 + self.args.prune_percent - cur_prune_rate)

            fprune_fixed_amount(self.model,
                               prune_fraction,#prune_step,
                               verbose=self.args.prune_verbosity)
        elif eval_score['Accuracy'][0] > -0.1 and cur_prune_rate < self.args.prune_percent:
            self.args.acc_thresh /= 2
            print("acc thresh dec to: ", self.args.acc_thresh, flush=True)
        elif eval_score['Accuracy'][0] < 0 and cur_prune_rate < self.args.prune_percent:
            eval_flag = 1
            

        losses = []
        accuracies = []

        for i in range(self.args.client_epoch):
            train_log_path = f'./log/clients/client{self.client_id}'\
                             f'/round{self.elapsed_comm_rounds}/'


            #print(f'Epoch {i+1}')
            train_score = ftrain(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            fprune_fixed_amount(self.model,
                               prune_fraction,#prune_step,
                               verbose=self.args.prune_verbosity)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
            epoch_path = train_log_path + f'client_model_epoch{i}.torch'
            epoch_score_path = train_log_path + f'client_train_score_epoch{i}.pickle'
            log_obj(epoch_path, self.model)
            log_obj(epoch_score_path, train_score)

            
        if eval_flag == 1:
            fprune_fixed_amount(self.model,
                               0.1,#prune_step,
                               verbose=self.args.prune_verbosity)
        mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")

        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        self.prune_rates[round_index:] = cur_prune_rate

        #self.prune_rates.append(self.curr_prune_rate)

        return copy_model(self.model, self.args.dataset, self.args.arch)

    def test_client_update_save(self, global_model, global_init_model, round_index):
        self.elapsed_comm_rounds += 1
        print(f'$$$$$ Client #{self.client_id} $$$$$', flush=True)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        prune_fraction = cur_prune_rate
        self.model = copy_model(global_model,
                                self.args.dataset,
                                #self.args.arch, dict(self.model.named_buffers()))
                                self.args.arch, dict(global_model.named_buffers()))
        for mod in list(self.model.modules()):
            if type(mod) != nn.Sequential and type(mod) != type(self.model):
                for name, param in mod.named_parameters():
                    if 'weight' in name: 
                        prune.remove(mod, 'weight')
        eval_flag = 0
        eval_score = fevaluate(self.model, 
                         self.test_loader,
                         verbose=self.args.test_verbosity)
        fprune_fixed_amount(self.model,
                               0,#prune_step,
                               verbose=self.args.prune_verbosity)

        eval_log_path = f'./log/clients/client{self.client_id}/'\
                        f'round{self.elapsed_comm_rounds}/'\
                        f'eval_score_round{self.elapsed_comm_rounds}.pickle'
        log_obj(eval_log_path, eval_score)


        if eval_score['Accuracy'][0] > self.args.acc_thresh and cur_prune_rate < self.args.prune_percent:
            self.args.acc_thresh = 0.5
            # I'm adding 0.001 just to ensure we go clear the target prune_percent. This may not be needed
            prune_fraction = cur_prune_rate + min(self.args.prune_step, 0.001 + self.args.prune_percent - cur_prune_rate)

            fprune_fixed_amount(self.model,
                               prune_fraction,#prune_step,
                               verbose=self.args.prune_verbosity)
        elif eval_score['Accuracy'][0] > 0.1 and cur_prune_rate < self.args.prune_percent:
            self.args.acc_thresh /= 2
            print("acc thresh dec to: ", self.args.acc_thresh, flush=True)
        elif eval_score['Accuracy'][0] < 0.1 and cur_prune_rate < self.args.prune_percent:
            eval_flag = 1
            

        losses = []
        accuracies = []

        for i in range(self.args.client_epoch):
            train_log_path = f'./log/clients/client{self.client_id}'\
                             f'/round{self.elapsed_comm_rounds}/'


            #print(f'Epoch {i+1}')
            train_score = ftrain(round_index, self.client_id, i, self.model,
                  self.train_loader, 
                  lr=self.args.lr,
                  verbose=self.args.train_verbosity)
            fprune_fixed_amount(self.model,
                               prune_fraction,#prune_step,
                               verbose=self.args.prune_verbosity)
            losses.append(train_score['Loss'][-1].data.item())
            accuracies.append(train_score['Accuracy'][-1])
            epoch_path = train_log_path + f'client_model_epoch{i}.torch'
            epoch_score_path = train_log_path + f'client_train_score_epoch{i}.pickle'
            log_obj(epoch_path, self.model)
            log_obj(epoch_score_path, train_score)

            
        if eval_flag == 1:
            fprune_fixed_amount(self.model,
                               .1,#prune_step,
                               verbose=self.args.prune_verbosity)
        mask_log_path = f'{self.args.log_folder}/round{round_index}/c{self.client_id}.mask'
        client_mask = dict(self.model.named_buffers())
        log_obj(mask_log_path, client_mask)

        num_pruned, num_params = get_prune_summary(self.model)
        cur_prune_rate = num_pruned / num_params
        prune_step = math.floor(num_params * self.args.prune_step)
        print(f"num_pruned {num_pruned}, num_params {num_params}, cur_prune_rate {cur_prune_rate}, prune_step: {prune_step}")

        self.losses[round_index:] = np.array(losses)
        self.accuracies[round_index:] = np.array(accuracies)
        self.prune_rates[round_index:] = cur_prune_rate

        #self.prune_rates.append(self.curr_prune_rate)

        return copy_model(self.model, self.args.dataset, self.args.arch)
    
    def evaluate(self):
        eval_score = evaluate(self.model,
                              self.test_loader,
                              verbose=self.args.test_verbosity)
        return eval_score['Accuracy'][-1]

    def fevaluate(self):
        eval_score = fevaluate(self.model,
                              self.test_loader,
                              verbose=self.args.test_verbosity)
        return eval_score['Accuracy'][-1]
    def get_mask(self):
        result = np.array(())
        for k, v in self.model.named_buffers():
            if 'weight_mask' in k:
                result = np.append(result, [v.data.numpy().reshape(-1)])
        return np.array(result)

    def get_class_counts(self, dataset):
        if dataset == 'train':
            ds = self.train_loader
        elif dataset == 'test':
            ds = self.test_loader
        else:
            raise Error('get_class_counts() - invalid value for parameter dataset: ', dataset)

        class_counts = {}
        for batch in ds:
            for label in batch[1]:
                if label.item() not in class_counts:
                    class_counts[label.item()] = 0
                class_counts[label.item()] += 1
        return class_counts

    