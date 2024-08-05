# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.autograd.variable as Variable
import os
from copy import deepcopy
import torch.optim as optim
from progress.bar import Bar
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
from lib.utils.data_utils import get_split_train_dataset, get_data_loaders, get_voc_data_loaders,get_mnist_data_loaders, get_cifar_data_loaders
from lib.utils.quantize_utils import quantize_model_to_same_bit, kmeans_update_model, kmeans_update_model_mix,quantize_model, quantize_model_mix_kernel_bit
from lib.utils.utils import load, store
from models.mobilenet import MobileNet
import torch.nn.functional as F
import copy
from models.utils import nms, get_region_boxes, bbox_iou


class QuantizeEnv:
    def __init__(self, model, pretrained_model, data, data_root, compress_ratio, args, n_data_worker=16,
                 batch_size=256, float_bit=32, quantize_bit = 8, is_model_pruned=False, export_model=False):
        """
        Init observation and implement following observation/state's update required data
        Args:
            model: user select pre-trained model
            pretrained_model: state of pre-trained model
            data: data set type
            data_root: data set root
            compress_ratio: preserve ratio of the model size
            args: user input/default options
            n_data_worker: n_data_worker when sample to train/val loader or train actor/critic network
            batch_size: batch size of sample
            float_bit: the bit of full precision float
            quantize_bit: the max bit for compressed model
            is_model_pruned: is the pre-trained model pruned, bool
        """
        # default setting
        self.compressable_layer_types = [nn.Conv2d, nn.Linear]

        # save options
        self.model = model
        self.model_for_measure = deepcopy(model)
        # create a new model for finetune
        # self.model_for_finetune = deepcopy(model)
        self.cur_ind = 0
        # todo: change from list to list of list
        self.strategy = []  # CP strategy
        self.in_strategy = []

        self.finetune_lr = args.finetune_lr
        self.optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.pretrained_model = pretrained_model
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.data_root = data_root
        self.compress_ratio = compress_ratio
        self.args = args
        self.is_model_pruned = is_model_pruned
        self.val_size = args.val_size
        self.train_size = args.train_size
        self.finetune_gamma = args.finetune_gamma
        self.finetune_flag = args.finetune_flag
        self.finetune_epoch = args.finetune_epoch

        # options for channel pruning
        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.alpha = args.alpha

        # options from args
        self.quantize_bit = args.quantize_bit
        self.lbound = args.lbound  # this force each layer at least keep lbound percentage layer
        self.rbound = args.rbound
        self.float_bit = float_bit * 1.
        self.quantize_bit = quantize_bit * 1.
        self.classifier_ratio = args.classifier_preserve_ratio
        # self.last_action = self.max_bit don't use this

        self.is_inception = args.arch.startswith('inception')
        self.is_imagenet = ('imagenet' in data)
        self.matrix = args.matrix
        self.export_model = export_model
        # sanity check
        assert self.compress_ratio <= 1, 'Error! You can not make achieve preserve_ratio great than 1!'
        # option for quantization
        self.quantify_opt = args.quantify_opt
        # init reward
        self.best_reward = -math.inf
        #self.done_mix = False
        self.iou_thresh = args.iou_thresh
        # prepare data for train/val
        self._init_data()

        # build indexs
        self._build_index()
        # extract information[I/O FM, parameters and flops] of each prune-able layer
        self._extract_layer_information()
        print("FINISH EXTRACT LAYER INFO")
        # init approach to get layer's weight size by layer idx
        self._get_weight_size()
        self.n_quantizable_layer = len(self.quantifiable_idx)
        # copy pretrained_model's state dict's keys exactly same with self.model
        self.model.load_state_dict(self.pretrained_model, strict=True)

        # build embedding (static part), same as pruning
        self._build_state_embedding()

        # restore weight, build reward
        self.best_ratio = 1.0
        self.best_compressed_acc = 0.0
        self.reset()
        if self.matrix == 'AP':
            self.best_AP = -math.inf
            self.data_sample_size = args.data_sample_size
            if self.data_type == 'VOC':
                self.org_AP = self._validate_voc(self.model, self.val_loader)
            else:
                if self.data_type == 'MNIST':
                    self.org_AP = self._validate_mnist(model=self.model, test_loader=self.val_loader)
                else:
                    if self.data_type == 'CIFAR':
                        self.org_AP = self._validate_cifar(net=self.model, testloader=self.val_loader)
                    else:  # pan or mul
                        self.org_AP = self._validate_det(self.val_loader, self.model)
            print('=> original AP: {:.3f} '.format(self.org_AP))
        else:
            self.best_acc = -math.inf
            self.use_top5 = False
            if self.matrix == 'top-5':
                self.use_top5 = True
            self.org_acc, loss= self._validate_reco(self.val_loader, self.model)
            print('=> original acc: {:.3f}% on split dataset(train: %7d, val: %7d )'.format(self.org_acc,
                                                                                            self.train_size, self.val_size))
        print('=> original #param: {:.4f}, model size: {:.4f} MB'.format(sum(self.wsize_list) * 1. / 1e6,

                                                                         sum(self.wsize_list) * self.float_bit / (self.quantize_bit * 1e6)))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.layer_info_dict[idx]['flops'] / 1e6 for idx in sorted(self.layer_info_dict.keys())])
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))
        self.best_strategy = None

        self.best_d_prime_list = None

        self.org_w_size = sum(self.org_wsize_list)
        # get expected parameter size, change here in future
        # change here to achieve higher compression bound , 4 - 2 -1
        self.expected_preserve_computation = 8 * self.compress_ratio * self.org_flops


    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma

    def update_kernel_state(self, importance, preserved, pruned_idx, action):
        """
            Update at each prunable layer for quantifaibale layer between [prev_prunable, curprunable layer]
            by importance and preserved
        """
        cur_idx_in_quantify = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
        if importance is None or pruned_idx is None:
            self.layer_can_select[cur_idx_in_quantify] = 0
            if importance is not None:
                self.layer_preserved[cur_idx_in_quantify] = len(importance)
            return

        def compute_min_bit_by_importance(importance, max_bit, pruned_idx):
            """
            compute kernel min bit by importance
            Args:
                importance: >0
                max_bit: user defined max bit after quantization
                preserve_idx: index of preserved
            Returns:

            """
            max_importance = np.max(importance)
            result = []
            max_bit_count = 0
            can_select_channel_num = 0
            for i in importance:
                cur_value = np.round((i * max_bit) / max_importance)
                if cur_value == 0:
                    cur_value = 1
                if cur_value >= max_bit:
                    cur_value = max_bit
                    max_bit_count += 1
                result.append(cur_value)
            result = np.asarray(result)
            if len(pruned_idx) > 0:
                result[pruned_idx] = 0
            can_select_channel_num = len(importance) - (len(pruned_idx) + max_bit_count)
            return result, can_select_channel_num

        # update current prunable layer first
        self.layer_importance[cur_idx_in_quantify] = importance
        self.layer_preserved[cur_idx_in_quantify] = preserved
        allowed_min_bit, can_select_channel_num = compute_min_bit_by_importance(importance, self.quantize_bit, pruned_idx)
        self.layer_allowed_min_bit[cur_idx_in_quantify] = allowed_min_bit
        self.layer_can_select[cur_idx_in_quantify] = can_select_channel_num
        #update former quantifable layer index if exist
        prev_idx_in_quantify = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind - 1])
        for idx in range(prev_idx_in_quantify + 1, cur_idx_in_quantify):
            self.layer_importance[idx] = importance
            self.layer_preserved[idx] = preserved
            self.layer_allowed_min_bit[idx] = allowed_min_bit
            self.layer_can_select[idx] = can_select_channel_num

    def estimate_divergence(self):
        return
    def step_cm_no_adjust(self, actions, importance_json=None):
        """
            Do channel pruning and quantization by
                1. get 9 action
                2. do channel pruning
                3. do quantization
        Args:
            action:

        Update: add quantization strategy

        """

        # action mean how many element will be preserved and quantized
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        if self.visited[self.cur_ind]:
            actions = self.strategy[self.cur_ind]
            preserve_idx = self.index_buffer[self.cur_ind]  # init at build index
        elif self.export_model:
            print("Use user input action in %f"%(self.cur_ind))
            preserve_idx = None
        else:
            actions = self._action_wall_mix(actions)  # percentage to preserve
            preserve_idx = None

        # prune and update action, change model of prune-able layer
        pruning_rate, preserve_idx, importance, pruned_idx, actions_ad = self.prune_kernel_mix(self.prunable_idx[self.cur_ind], actions, preserve_idx, importance_json)
        delta_w = self.estimate_divergence()
        actions = actions_ad
        if not self.visited[self.cur_ind]:
            for group in self.shared_idx:  # block has shared layer
                if self.cur_ind in group:  # set the shared ones
                    for g_idx in group:
                        self.strategy_dict[self.prunable_idx[g_idx]][0] = pruning_rate
                        self.strategy_dict[self.prunable_idx[g_idx - 1]][1] = pruning_rate
                        self.visited[g_idx] = True
                        self.index_buffer[g_idx] = preserve_idx.copy()

        if self.export_model:  # export checkpoint
            print('# Pruning {}: ratio: {}'.format(self.cur_ind, pruning_rate))

        self.strategy.append(actions)  # save action to strategy
        #self.d_prime_list.append(d_prime)
        # save input and output channel sparsity
        # current layer output is next layer input, cuz pruning input channel, update former layer output upper bound
        # cur_ind = 0, first layer should do action, let first prune-able layer do this
        self.strategy_dict[self.prunable_idx[self.cur_ind]][0] = pruning_rate
        if self.cur_ind > 0:
            self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = pruning_rate
        # all the actions are made
        if self._is_final_layer():
            # pseudo channel pruning pruning
            self.visited[self.cur_ind] = True
            #todo: update here by mix bit
            pruned_layer_weight = self.get_pruned_layer_weight()
            self.set_pruned_layer_weight(pruned_layer_weight)
            def computePlan(layer_kernel_bit):
                layer_QP = []
                kernel_QP = []
                model_quantify_strategy = []
                for i in range(len(layer_kernel_bit)):  # for each quantifiable layer
                    # check is all 8
                    cur_strategy = np.zeros(9).tolist()
                    isDP = np.all((layer_kernel_bit[i] - 8) == 0)
                    layer_QP.append(isDP)
                    if not isDP:  # i-th layer was selected to perform deep quantization
                        cur_layer_DP = []
                        for m in range(9):  # generate current layer DP for 8bit
                            cur_layer_DP.append([])
                        for j in range(len(layer_kernel_bit[i])):  # for each channel
                            index = int(layer_kernel_bit[i][j])
                            cur_layer_DP[index].append(j)
                        kernel_QP.append(cur_layer_DP)
                        for m in range(9):
                            cur_strategy[m] = len(cur_layer_DP[m])
                    else:  # not selected
                        kernel_QP.append([8])
                        cur_strategy[8] = len(layer_kernel_bit[i])
                    model_quantify_strategy.append(cur_strategy)
                #write model_quantify_strategy to txt
                # ([length of quantiziable layer],[9])

                return layer_QP, kernel_QP, model_quantify_strategy

            #prev_acc, prev_loss = self._validate_reco(self.val_loader, self.model)
            #print(prev_acc, prev_loss)
            layer_QP, kernel_QP, model_quantify_strategy = computePlan(self.layer_kernel_bit)

            if self.export_model:

                torch.save(self.model.state_dict(), self.export_path)
                return None, None, None, None
            #acc, loss = self._validate_reco(self.val_loader, self.model)
            #print(acc, loss)
            centroid_label_dict = quantize_model_mix_kernel_bit(self.model, self.quantifiable_idx, self.prunable_idx, layer_QP, kernel_QP,
                                          self.quantize_bit, is_pruned=True, max_iter=3)
            #acc, loss = self._validate_reco(self.val_loader, self.model)
            #print(acc, loss)
            w_size, element_bit_by_layer, total_nz = self._get_mix_precission_size(layer_QP, kernel_QP)
            w_size_ratio = w_size / (self.org_w_size * 32)
            flops_ratio = self._cur_flops() / self.org_flops
            print(
                'cur policy (avg bit): {}\n, w_ratio: {:.3f}, flops_ratio:{:.3f}, NZ%{:.3F}'.format(
                    self.avg_wsize_list, w_size_ratio, flops_ratio, total_nz/self.org_w_size))
            # self._final_action_wall()
            assert len(self.strategy) == len(self.prunable_idx)

            if w_size_ratio < self.best_ratio:
                self.best_ratio = w_size_ratio
                if self.matrix == 'AP':
                    if self.data_type == 'VOC':
                        acc = self._validate_voc(self.model, self.val_loader)

                    else:
                        if self.data_type == 'MNIST':
                            acc = self._validate_mnist(model=self.model, test_loader=self.val_loader)
                        else:
                            if self.data_type == 'CIFAR':
                                acc = self._validate_cifar(net=self.model, testloader=self.val_loader)
                            else:# pan or mul
                                acc = self._validate_det(self.val_loader, self.model)
                else:# imagenet
                    if self.finetune_flag and self.matrix == 'top-5':
                        train_acc = self._kmeans_finetune(self.train_loader, self.model, self.quantifiable_idx,
                                                          centroid_label_dict, epochs=self.finetune_epoch,
                                                          verbose=False, layer_QP=layer_QP, kernel_QP=kernel_QP)
                        w_size_ratio = w_size / (self.org_w_size * 32)
                        flops_ratio = self._cur_flops() / self.org_flops
                        layer_QP = None
                    acc,loss = self._validate_reco(self.val_loader, self.model)
                    print(acc, loss)
                self.best_compressed_acc = acc
            else:
                if flops_ratio == 1.:
                    acc, loss = self._validate_reco(self.val_loader, self.model)
                else:
                    acc = self.best_compressed_acc
            info_set = {'w_ratio': w_size_ratio, 'FLOPS_ratio': flops_ratio, 'accuracy': acc, 'w_size': w_size}
            if w_size_ratio <= self.compress_ratio:
                reward = self.reward(acc)
            else:
                reward = self.size_bound_reward(w_size_ratio)

            if reward > self.best_reward:
                # worth to quantify, save more space with same acc
                self.best_reward = reward
                self.best_acc = acc
                import csv
                import datetime
                csv_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.csv'
                path = './' + csv_filename
                with open(path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for layer_strategy in model_quantify_strategy:  # for each quantizable layer
                        writer.writerow(layer_strategy)

                if not layer_QP is None:
                    performed_layers = [i for i in range(len(layer_QP)) if layer_QP[i] == False]
                    print('=> Deep quantization profermed layers: {}'.format(performed_layers))
                    print('=> Deep quantization profermed layers: {}'.format(element_bit_by_layer))

                prGreen(
                    'New best policy (avg bit): {}, reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}, flops_ratio:{:.3f}'.format(
                        self.avg_wsize_list, self.best_reward, acc, w_size_ratio, flops_ratio))
                prGreen(
                    'New best policy (pruning rate): {}'.format(
                        self.strategy_dict))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set


        done = False
        self.visited[self.cur_ind] = True  # set to visited

        #todo: update here, update weight by mixed actions
        # pruned in group conv 1x1 layer
        pruned_layer_weight = self.get_pruned_layer_weight()
        self.set_pruned_layer_weight(pruned_layer_weight)
        # calculate weight(MB) so far
        w_size = self._cur_weight()
        w_size_bit = self._max_weight_size_so_far()

        reward = self.all_layer_reward(delta_w, w_ratio, acc)
        info_set = {'w_size': w_size}
        self.cur_ind += 1  # the index of next prunable layer
        # todo: build next state (in-place modify), should update to  add parameter size changed
        # current strategy reduced flops and strategy uncovered part

        self.layer_embedding[self.cur_ind][-12] = self._cur_reduced() * 1. / self.org_flops  # strategy reduced flops
        self.layer_embedding[self.cur_ind][-11] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # strategy rest flops
        # current strategy reduced weight(bit) and strategy uncovered part
        #todo: change from parameter num to parameter size
        self.layer_embedding[self.cur_ind][-10] = self._reduced_weight(w_size) * 1. / self.org_w_size * 32
        self.layer_embedding[self.cur_ind][-9] = self._rest_weight() * 1. / self.org_w_size * 32
        #self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # todo: change from last pruning rate to actions
        # todo: here update action to actions
        cur_strategy = self.strategy[-1]
        for i in range(9):
            if i == 0:
                self.layer_embedding[self.cur_ind, -(i + 1)] = cur_strategy[-(i + 1)]
            else:
                self.layer_embedding[self.cur_ind, -(i + 1)] = cur_strategy[-(i + 1)]

        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        return obs, reward, done, info_set


       # for quantization
    def reward(self, acc):
        if self.matrix == 'AP':
            return (acc - self.org_AP)* 0.1
        return (acc - self.org_acc) * 0.1

    # ----------------REWAED: self defined-----------------------------
    def size_bound_reward(self, w_ratio):
        if w_ratio > self.compress_ratio:
            diff = w_ratio - (self.compress_ratio)
            return -(10. + diff*10)
        return 0.




    def multibounds_reward(self, trained_acc, acc, best_acc, w_ratio):
        """
        Penalty a lot when can't meet compression target
        Args:
            target_size: user required network size(param size)
            cur_size: CPQA light-weighted network size(param size)
            acc: CPQA light-weighted model accuracy
            trained_acc: fine tuned acc
        Returns: reward
        Update:
            use flops to represent compression of model function
        """
        if w_ratio > self.compress_ratio :
            return -10.
        if trained_acc > best_acc:
            return (acc - self.org_acc) * 0.1 + (trained_acc - acc) * 0.05
        else:
            return(acc - self.org_acc) * 0.1

    def ratio_related_reward(self, w_ratio, flops_ratio, acc):
        """
        find flops decrease can make more w_size decline while retain higher acc
        Args:
            w_ratio: reduced_weight_size/org_weight_size
            flops_ratio:reduced_flops / org_flops
            acc: acc after quantization
        Returns:

        """
        if w_ratio > self.compress_ratio:
            return -10.
        return (acc *(flops_ratio / w_ratio)) * 0.1

    def all_layer_reward(self, delta_w, w_ratio, acc):
        if self._is_final_layer():
            if w_ratio > self.compress_ratio:
                return -10.
            return acc - self.org_acc
        else:
            if delta_w < self.alpha:
                return 0
            return -delta_w

    def get_pruned_layer_weight(self):
        """
        get pruned layer weight, assumed that current layer is pruned
        Returns:
            pruned layer weight (in count)
        """
        assert self.visited[self.cur_ind], 'ERROR: Only call get_pruned_layer_weight after cur_ind-th layer is pruned'
        cur_layer_weight = 0
        for i, m in enumerate(self.model.modules()):
            if i == self.prunable_idx[self.cur_ind]:
                # compute number of element not equal to 0
                nz_mask = m.weight.data.ne(0)
                cur_layer_weight += torch.sum(nz_mask) # count number of element in m.weight.data is numel()

                break
        return cur_layer_weight

    def get_quantized_model_weight(self, isFinetune = False):
        """
        get pruned model weight(in number) after the whole model is quantized
        Returns:

        """
        quantized_model_weight = 0
        """
        if isFinetune:
            model = self.model_for_finetune
        else:
        """
        model = self.model
        for i, m in enumerate(model.modules()):
            if i in self.quantifiable_idx:
                nz_mask = m.weight.data.ne(0)
                quantized_model_weight += torch.sum(nz_mask)
        quantized_model_weight = quantized_model_weight.item()
        return quantized_model_weight

    def _get_mix_precission_size(self, layer_QP, kernel_QP):
        quantized_model_size = 0
        """
        if isFinetune:
            model = self.model_for_finetune
        else:
        """
        model = self.model
        bit_each_element_by_layer = []
        totoal_nz = 0
        if kernel_QP is None:
            for i, m in enumerate(model.modules()):
                if i in self.quantifiable_idx:
                    idx = self.quantifiable_idx.index(i)
                    nz_mask = m.weight.data.ne(0)
                    current_weight = torch.sum(nz_mask)
                    quantized_model_size += current_weight.item() * self.quantization_strategy[idx]
                    totoal_nz += current_weight.item()
        else:
            for i, m in enumerate(model.modules()):
                if i in self.quantifiable_idx:
                    idx_quantify = self.quantifiable_idx.index(i)
                    cur_kernels = kernel_QP[idx_quantify]
                    cur_w = m.weight.data
                    if layer_QP[idx_quantify]:
                        nz_mask = cur_w.ne(0)
                        current_weight = torch.sum(nz_mask)
                        quantized_model_size += current_weight.item() * self.quantize_bit
                        totoal_nz += current_weight.item()
                        bit_each_element_by_layer.append(self.quantize_bit)
                    else:
                        cur_total_weight = 0
                        cur_total_size = 0
                        cur_total_nz = 0
                        if i in self.prunable_idx:
                            for bit in range(len(cur_kernels)):
                                if len(cur_kernels[bit]) == 0:
                                    continue
                                cur_sub_w = cur_w[:, cur_kernels[bit]]
                                nz_mask = cur_sub_w.ne(0)
                                current_sub_weight = torch.sum(nz_mask)
                                cur_total_weight += current_sub_weight
                                cur_total_size += current_sub_weight.item() * (bit)
                                cur_total_nz += current_sub_weight.item()
                            quantized_model_size += cur_total_size
                            avg_bit = cur_total_size / cur_total_weight
                            totoal_nz += cur_total_nz
                            bit_each_element_by_layer.append(avg_bit.item())
                        else:
                            """
                            total_bit = 0
                            channels = 0
                            for bit in range(1, len(cur_kernels)):
                                cur_channels = len(cur_kernels[bit])
                                channels += cur_channels
                                total_bit += bit * cur_channels
                            avg_bit = total_bit / channels
                            n,c,k,k = cur_w.size()
                            quantized_model_size += n * c * k * k * avg_bit
                            bit_each_element_by_layer.append(avg_bit)
                            """
                            cur_total_weight = 0
                            cur_total_size = 0
                            cur_total_nz = 0
                            for bit in range(len(cur_kernels)):
                                if len(cur_kernels[bit]) == 0:
                                    continue
                                cur_sub_w = cur_w[ cur_kernels[bit],:]
                                nz_mask = cur_sub_w.ne(0)
                                current_sub_weight = torch.sum(nz_mask)
                                cur_total_weight += current_sub_weight
                                cur_total_size += current_sub_weight.item() * (bit)
                                cur_total_nz += current_sub_weight.item()
                            quantized_model_size += cur_total_size
                            avg_bit = cur_total_size / cur_total_weight
                            totoal_nz += cur_total_nz
                            bit_each_element_by_layer.append(avg_bit.item())
        return quantized_model_size, bit_each_element_by_layer, totoal_nz


    def set_pruned_layer_weight(self, layer_weight):
        """
        Update cur_layer_weight list and dict
        Args:
            layer_weight: tensor of pruned layer weight

        Returns:
            null
        """
        # get value from layer_weight tensor
        weight_value = layer_weight.item()
        index_in_Q = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
        self.wsize_list[index_in_Q] = weight_value
        self.wsize_dict[self.quantifiable_idx[index_in_Q]] = weight_value

    def _cur_weight(self):
        """
        current weight(count) so far
        Attention: self.wsize_list may have bug to cuz 0
        Returns:

        """
        cur_weight = 0.
        quantization_idx = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
        for i in range(quantization_idx + 1):
            cur_weight += self.wsize_list[i]
        return cur_weight

    def _cur_weight_size(self):
        cur_weight_size = 0.
        quantization_idx = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
        for i in range(quantization_idx + 1):
            cur_weight_size += self.wsize_list[i] * self.avg_wsize_list[i]
        return cur_weight_size

    def _cur_weight_size(self):
        cur_weight_size = 0.
        for i in range(len(self.quantization_strategy)):
            cur_weight_size += self.wsize_list[i] * self.quantization_strategy[i]
        return cur_weight_size

    def _max_weight_size_so_far(self):
        cur_weight = 0.
        quantization_idx = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
        for i in range(len(self.quantization_strategy)):
            if i < quantization_idx:
                cur_weight += self.wsize_list[i] * self.avg_wsize_list[i]
            else:
                cur_weight += self.wsize_list[i] * self.quantize_bit
        return cur_weight

    def _reduced_weight(self, cur_weight):
        """
        cur_weight : covered model weight in bit
        Get reduced weight (in num) of current strategy
        reduced = original_strategy_covered - strategy_covered
        Returns:
        reduced weight in bit
        """
        strategy_covered_index = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
        # compute original params size
        #todo: update _reduced_weight in bit
        org_params = sum(self.org_wsize_list[: strategy_covered_index + 1]) * 32
        reduced_weight = org_params - cur_weight
        return reduced_weight

    def _rest_weight(self):
        """
        original_strategy_uncovered layer
        Returns:
            rest weight in bit

        """
        current_layer_index = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
        rest_weight = sum(self.org_wsize_list[current_layer_index + 1:]) * 32
        return rest_weight

        print("not implement")

    # ----------------ENV--------------------------------

    def reset(self):
        """
        Reset from pretrained model(alexnet, vgg) or check point(resnet)[not implemented yet]
        Returns:

        """
        # restore env by loading the pretrained model
        self.model.load_state_dict(self.pretrained_model, strict=False)
        #self.model_for_finetune = deepcopy(self.model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=4e-5)

        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        self.layer_kernel_bit = copy.deepcopy(self.org_layer_kernel_bit)
        def set_quantization_strategy(strategy, quantize_bit):
            for i in range(len(strategy)):
                strategy[i] = quantize_bit
        set_quantization_strategy(self.quantization_strategy, self.quantize_bit)
        # for channel pruning
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)
        #self.d_prime_list = []
        # reset layer embeddings
        #todo: here update action to actions
        for i in range(9):
            if i == 0:
                self.layer_embedding[:, -(i+1)] = 1.
            else:
                self.layer_embedding[:, -(i+1)] = 0.
        #self.layer_embedding[:, -1] = 1. [0., 0., 0., 0., 0., 0., 0., 0., 1.]  # action
        self.layer_embedding[:, -9] = 0.  # param rest
        self.layer_embedding[:, -10] = 0.  # param reduced
        self.layer_embedding[:, -11] = 0.  # flops rest
        self.layer_embedding[:, -12] = 0.  # flops reduced
        obs = self.layer_embedding[0].copy()
        # use org_wsize_list, cuz wsize has been changed during channel pruning
        obs[-2] = sum(self.org_wsize_list[1:]) * 1. / sum(self.org_wsize_list)
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0
        # for share index
        self.visited = [False] * len(self.quantifiable_idx)
        self.index_buffer = {}
        return obs

    def _is_final_layer_out(self):
        return self.cur_ind == len(self.prunable_idx) - 2

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    # ------------------------!Self Defined function _light_weight_action!----------------------#
    def _isFC(self):
        return self.prunable_idx[self.cur_ind] in self.FC_idx

    def _action_wall_mix(self, actions):
        """
        generate a legal action,range in [lbound, rbound], return 1. when current layer is not prune-able
        Args:
            actions: 1d list in length of 9,random generated numbers range in [0,1]

        Returns:
            actions: 1d list in length of 9, bounded by flops or pre-defined searching range
        """

        assert len(self.strategy) == self.cur_ind
        # limit the action to certain range
        def convert_each_action(action, bit_idx):

            return action

        def computePlan(layer_kernel_bit):
            layer_QP = []
            kernel_QP = []
            for i in range(len(layer_kernel_bit)):  # for each quantifiable layer
                # check is all 8
                isDP = np.all((layer_kernel_bit[i] - 8) == 0)
                layer_QP.append(isDP)
                if not isDP:  # i-th layer was selected to perform deep quantization
                    cur_layer_DP = []
                    for m in range(9):  # generate current layer DP for 8bit
                        cur_layer_DP.append([])
                    for j in range(len(layer_kernel_bit[i])):  # for each channel
                        index = int(layer_kernel_bit[i][j])
                        cur_layer_DP[index].append(j)
                    kernel_QP.append(cur_layer_DP)
                else:  # not selected
                    kernel_QP.append([8])
            return layer_QP, kernel_QP
        w_size = self.org_w_size * 32
        cur_channelnum = self.org_channels[self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])]
        if self.args.job == 'train':
            layer_QP, kernel_QP = computePlan(self.layer_kernel_bit)
            w_size, element_bit_by_layer, total_nz = self._get_mix_precission_size(layer_QP, kernel_QP)
        if w_size / (self.org_w_size * 32) < self.compress_ratio or self.cur_ind == 0:
            actions = np.zeros(len(actions))
            actions[-1] = 1.
            return actions
        else:
            other_comp = 0
            this_comp = 0
            prune_index = self.cur_ind
            for i, idx in enumerate(self.prunable_idx):
                flop = self.layer_info_dict[idx]['flops']
                buffer_flop = 0
                buffer_flop = self._get_buffer_flops(idx)
                # self.cur_ind meaning? cur_ind-th prune-able layer
                if i == prune_index - 1:  # TODO: add other member, current in the layer before a pruned layer
                    this_comp += flop * (1 - self.strategy_dict[idx][0])  # prune dependent part
                    # add buffer (but not influenced by ratio)
                    other_comp += buffer_flop * (1 - self.strategy_dict[idx][0])
                elif i == prune_index:  # cur_id is ith prune-able layer, do action
                    this_comp += flop * (1 - self.strategy_dict[idx][1])
                    # also add buffer here (influenced by ratio)
                    this_comp += buffer_flop
                else:  # current in the layer after a pruned layer, computation decrease due to current pruning
                    other_comp += flop * (1 - self.strategy_dict[idx][0]) * (1 - self.strategy_dict[idx][1])  # reduction in FC, 0 for input 1 for output
                    # add buffer
                    other_comp += buffer_flop * (1 - self.strategy_dict[idx][0])  # only consider input reduction

            max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp
            min_pruning_ratio = 1 - max_preserve_ratio
            actions[0] = np.maximum(actions[0], min_pruning_ratio)
            actions[0] = np.minimum(actions[0], self.rbound)
            total_bit = 0
            for j in range(8):
                total_bit += actions[j+1] * (j+1)
            avg_bit = (total_bit * cur_channelnum) / (int(cur_channelnum*(1-actions[0])))
            actions[1:] = (1. - actions[0]) * (actions[1:] / np.sum(actions[1:]))
            expected_avg_bit = (32 * self.compress_ratio)/((1. - actions[0]) ** 2)
            if avg_bit > expected_avg_bit:
                #adjust actions
                sorted_actions = deepcopy(actions[1:])
                re_sort_idx = np.sort(sorted_actions)
                alpha = 2
                for p in range(8):
                    actions[-(p+1)] = re_sort_idx[p] * (p + 1) * alpha
                actions[1:] = (1. - actions[0]) * (actions[1:] / np.sum(actions[1:]))
            return actions

    def _get_buffer_flops(self, idx):
        """
        get buffer flops
        Args:
            idx:

        Returns:

        """
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _get_buffer_params(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_params = sum([self.layer_info_dict[_]['params'] for _ in buffer_idx])
        return buffer_params

    def _cur_flops(self):
        """
        get flops of strategy, only effect when all layers are visited, for unvisited layer, c, n should be 1
        Returns:

        """
        flops = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruned ratio
            flops += self.layer_info_dict[idx]['flops'] * (1. - c) * (1. - n)
            # add buffer computation
            flops += self._get_buffer_flops(idx) * (1. - c)  # only related to input channel reduction
        return flops


    def set_export_path(self):
        """
        set export_path to user defined path
        """
        self.export_path = './weights/mobilenet_0.028size_export_imagenet.pth.tar'

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _org_weight(self):
        org_weight = 0.
        org_weight += sum(self.org_wsize_list) * self.float_bit
        return org_weight

    def _init_data(self):
        """
            init train and validation data loader for following quantization fine-tuning
            or performance evaluation
        """
        if self.matrix != 'AP': #imagenet
            self.train_loader, self.val_loader, n_class = get_split_train_dataset(
                self.data_type, self.batch_size, self.n_data_worker, data_root=self.data_root,
                val_size=self.val_size, train_size=self.train_size, for_inception=self.is_inception)
        else:
            if self.data_type == 'VOC':
                self.train_loader, self.val_loader, N = get_voc_data_loaders(pascal_path=self.data_root, batch=self.batch_size,
                                                                          num_workers = self.n_data_worker, random_crops = 10)
            else:
                if self.data_type == 'CIFAR':
                    self.train_loader, self.val_loader = get_cifar_data_loaders(self.batch_size)
                else:
                    if self.data_type == 'MNIST':
                        self.train_loader, self.val_loader = get_mnist_data_loaders(self.batch_size)
                    else: # pan or mul
                        self.train_loader, self.val_loader= get_data_loaders(self.data_root, self.batch_size, self.n_data_worker)


    def _build_index(self):
        """
        init quantifiable related info
        provide approach to get prune-abel layer by layerindex
            bound each prune-able layer (except first layer) at least prune lbound
        same as AMC, except save layers' type in list
        input:
            self.model: given by user, indicated as network name
            self.min_bit/self.max_bit: given by user, default as 0 / 8
        Returns:
            init quantifiable related info: quantifiable layer index, quantifiable layer type
        UPDATE:
        ADD QUANTIZATION STARTEGY
        """
        # for channel pruning
        self.prunable_idx = []
        self.prunable_ops = []
        self.layer_type_list = []
        self.strategy_dict = {}
        this_buffer_list = []
        self.org_channels = []
        self.org_in_channels=[]
        self.org_out_channels = []
        # for quantization
        self.quantifiable_idx = []
        self.quantization_strategy = []
        self.buffer_dict = {}
        self.FC_idx =[]
        for i, m in enumerate(self.model.modules()):
            #print(i, '->', m)
            if type(m) in self.compressable_layer_types:
                # depth-wise separable layer
                if type(m) == nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, buffer
                    this_buffer_list.append(i)
                    # self.strategy_dict[i] = [self.rbound, self.rbound] # don't do channel pruning in this layer
                else:  # really prunable
                    self.prunable_idx.append(i)
                    self.org_out_channels.append(m.out_channels if type(m) == nn.Conv2d else m.out_features)
                    self.org_in_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features)
                    self.prunable_ops.append(m)
                    # contain all residual block elements without connect layer for a specific residual block
                    self.buffer_dict[i] = this_buffer_list
                    this_buffer_list = []  # empty for next residual block
                    # todo: org_channels means all compressed layers not prunable layers
                    if type(m) == nn.Conv2d:
                        self.strategy_dict[i] = [self.lbound, self.lbound] # change to lbound, then can be update
                    else:
                        self.strategy_dict[i] = [self.classifier_ratio, self.classifier_ratio]
                        self.FC_idx.append(i)
                self.org_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features)
                self.layer_type_list.append(type(m))
                # add quantifiable index
                self.quantifiable_idx.append(i)
                self.quantization_strategy.append(self.quantize_bit)

        # first layer, at least prune 1 * 100 %, which means don't prune first layer input channel
        self.strategy_dict[self.prunable_idx[0]][0] = 0.  # modify the input
        # last layer, at most prune 1 * 100 %, which means don't prune last layer output channel
        self.strategy_dict[self.prunable_idx[-1]][1] = 0.  # modify the output

        # indicators of residual blocks
        self.shared_idx = []
        # change from model to arch
        if self.args.arch == 'mobilenetv2':  # TODO: to be tested! Share index for residual connection
            # all projection layer, after projection is an add operation
            connected_idx = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # to be partitioned
            last_ch = -1
            share_group = None
            for c_idx in connected_idx:
                if self.prunable_ops[c_idx].in_channels != last_ch:  # new group
                    last_ch = self.prunable_ops[c_idx].in_channels
                    if share_group is not None:
                        self.shared_idx.append(share_group)
                    share_group = [c_idx]
                else:  # same group
                    share_group.append(c_idx)
            print('=> Conv layers to share channels: {}'.format(self.shared_idx))
        if self.args.arch == 'resnet50':  # TODO: to be tested! Share index for residual connection
            # all projection layer, after projection is an add operation
            connected_idx = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]  # to be partitioned
            last_ch = -1
            share_group = None
            for c_idx in connected_idx:
                if self.prunable_ops[c_idx].in_channels != last_ch:  # new group
                    last_ch = self.prunable_ops[c_idx].in_channels
                    if share_group is not None:
                        self.shared_idx.append(share_group)
                    share_group = [c_idx]
                else:  # same group
                    share_group.append(c_idx)
            print('=> Conv layers to share channels: {}'.format(self.shared_idx))
        # bound each layer at least prune lbound
        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

        self.buffer_idx = []
        for k, v in self.buffer_dict.items():
            self.buffer_idx += v

        print('=> Prunable layer idx: {}'.format(self.prunable_idx))
        print('=> Buffer layer idx: {}'.format(self.buffer_idx))
        print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

        # added for supporting residual connections during pruning
        # set all compress-able layer to unvisited
        self.visited = [False] * len(self.quantifiable_idx)
        # memory for former strategy: former preserved index
        self.index_buffer = {}

    def _get_weight_size(self):
        """
        get the param size for each layers to prune and quantifiable, size expressed in number of params
        Returns:
            number of params for each quantifiable layer in self.wsize_list
            dict to access quantifiable layer by layer index
            Update:
            dict to access quantifiable layer's out, in and kernel by index
        """
        self.wsize_list = []
        #todo: add avg here
        self.avg_wsize_list = []
        self.org_wsize_list = []
        self.wshape_list = []
        self.layer_size = []
        self.layer_can_select = []
        self.layer_importance = []
        self.layer_preserved = []
        self.layer_kernel_bit = []
        self.layer_allowed_min_bit = []
        self.layer_kernel_changed_size = []
        self.org_layer_kernel_bit = []
        for i, m in enumerate(self.model.modules()):
            if i in self.quantifiable_idx:
                weight = m.weight.data.cpu().numpy()
                cur_shape = []
                if type(m) == nn.Conv2d:
                    cur_shape.append(m.out_channels)
                    cur_shape.append(m.in_channels)
                    cur_shape.append(m.kernel_size)
                    n, c, k1, k2 = weight.shape
                else:
                    cur_shape.append('linear')
                    n, c= weight.shape
                    k1 = 1
                    k2 = 1
                    cur_shape.append(m.out_features)
                    cur_shape.append(m.in_features)
                self.wshape_list.append(cur_shape)
                self.layer_preserved.append(0)
                if not self.is_model_pruned:
                    # numel will give number of element of a tensor
                    cur_layer_params = m.weight.data.numel()
                    self.wsize_list.append(cur_layer_params)
                    self.layer_size.append(cur_layer_params * self.quantize_bit)
                    self.org_wsize_list.append(cur_layer_params)
                else:  # the model is pruned, only consider non-zeros items
                    cur_layer_params = torch.sum(m.weight.data.ne(0))
                    self.wsize_list.append(cur_layer_params)
                    self.layer_size.append(cur_layer_params * self.quantize_bit)
                    self.org_wsize_list.append(cur_layer_params)
                if i in self.prunable_idx:
                    channels = c
                    self.layer_kernel_changed_size.append(n * k1 * k2)
                else:
                    channels = n
                    self.layer_kernel_changed_size.append(c * k1 * k2)
                self.avg_wsize_list.append(self.quantize_bit)
                self.layer_can_select.append(channels)
                self.layer_kernel_bit.append(np.ones(channels) * self.quantize_bit)
                self.org_layer_kernel_bit.append(np.ones(channels) * self.quantize_bit)
                self.layer_allowed_min_bit.append(np.ones(channels))
                self.layer_importance.append(np.ones(channels) * 1.)
        # self.wsize_dict contains[(id, layer_weight_size)]
        self.org_wsize_dict = {i: s for i, s in zip(self.quantifiable_idx, self.org_wsize_list)}
        self.wsize_dict = {i: s for i, s in zip(self.quantifiable_idx, self.wsize_list)}
        self.wshape_dict = {i: s for i, s in zip(self.quantifiable_idx, self.wshape_list)}
        self.avg_wsize_dict = {i: s for i, s in zip(self.quantifiable_idx, self.avg_wsize_list)}

    def _get_latency_list(self):
        # use simulator to get the latency
        raise NotImplementedError

    def _get_energy_list(self):
        # use simulator to get the energy
        raise NotImplementedError

    def _build_state_embedding(self):
        """
        build observation for each prune-able layer
        Inputs: self.model_for_measure, model to quantization, input by user
        Returns:

        """
        if self.is_imagenet:
            measure_model(self.model_for_measure, 224, 224)
        else:  # measure model for cifar 32x32 input
            if self.data_type == 'VOC':
                measure_model(self.model_for_measure, 227, 227)
            else:
                measure_model(self.model_for_measure, 32, 32, self.data_type)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        # real_module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w * m.in_h])  # input feature_map_size
            elif type(m) == nn.Linear:
                this_state.append([0.])  # layer type, 0 for fc
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w * m.in_h])  # input feature_map_size

            this_state.append([i])  # index
            this_state.append(0.)  # reduced flops, 0 for init
            this_state.append(0.)  # rest flops, 0 for init
            this_state.append(0.)  # reduced weights, 0 for init
            this_state.append(0.)  # rest weights, 0 for in {t-1}
            #todo: change from a_{t-1} to actions in {}
            this_state.append([0., 0., 0., 0., 0., 0., 0., 0., 1.0])  # a_{t-1}
            layer_embedding.append(np.hstack(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    #todo: not necessary in current version
    def _kmeans_finetune(self, train_loader, model, idx, centroid_label_dict, layer_QP=None, kernel_QP=None, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_acc = 0.

        # switch to train mode
        model.train()
        end = time.time()
        t1 = time.time()
        bar = Bar('train:', max=len(train_loader))
        for epoch in range(epochs):
            #print(epoch)
            for i, (inputs, targets) in enumerate(train_loader):
                #print('{:2d}/{:2d}'.format(i, len(train_loader)))
                input_var, target_var = inputs.cuda(), targets.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient
                self.optimizer.zero_grad()
                loss.backward()

                # do SGD step
                self.optimizer.step()
                if self.quantify_opt == 'fix':
                    kmeans_update_model(model, self.quantifiable_idx, centroid_label_dict, free_high_bit=True)
                else:
                    kmeans_update_model_mix(model, self.quantifiable_idx, self.prunable_idx, layer_QP, kernel_QP,centroid_label_dict, free_high_bit=True)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()

            if self.use_top5:
                if top5.avg > best_acc:
                    best_acc = top5.avg
            else:
                if top1.avg > best_acc:
                    best_acc = top1.avg
            self.adjust_learning_rate()
        t2 = time.time()

        print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2 - t1))
        return best_acc

    def _validate_reco(self, val_loader, model, verbose=True):
        """
        do validation of quantified model
        Args:
            val_loader:
            model:
            verbose:

        Returns:

        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t1 = time.time()
        with torch.no_grad():
            # switch to evaluate mode
            m_list = list(self.model.modules())

            model.eval()

            end = time.time()
            bar = Bar('valid:', max=len(val_loader))
            for i, (inputs, targets) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                input_var, target_var = inputs.cuda(), targets.cuda()

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2 - t1))
        if self.use_top5:
            return top5.avg , losses.avg
        else:
            return top1.avg , losses.avg

    def _validate_det(self, val_loader, model, use_cuda=True):
        """
         val_loader：所有验证集的数据和标签
         model：训练好的模型
         iou_thresh：预测框与真实框的iou高于该值时认为检测正确
         batch_num：随机取样batch_num个batch，并计算每个batch的ap
         """

        def truths_length(truths):
            for i in range(100):
                if truths[i][1] == 0:
                    return i

        anchors = [0.8, 0.24, 0.32, 0.16, 1.36, 0.6, 0.4, 0.4]  # 测试的时候要改这个地方，每个数据集会对应一种anchors
        num_anchors = 4
        eps = 1e-5  # 防止分母为0
        conf_thresh = 0.2  # 过滤所有框中置信度低于该值的,为了检测出更多的船，这里设置的比较低
        nms_thresh = 0.01  # nms中设置iou阈值，即与最高置信度框iou为该值以上的框去掉

        def voc_ap(rec, prec, use_07_metric=False):
            """ ap = voc_ap(rec, prec, [use_07_metric])
            Compute VOC AP given precision and recall.
            If use_07_metric is true, uses the
            VOC 07 11 point method (default:False).
            """
            if use_07_metric:
                # 11 point metric
                ap = 0.
                for t in np.arange(0., 1.1, 0.1):
                    if np.sum(rec >= t) == 0:
                        p = 0
                    else:
                        p = np.max(prec[rec >= t])
                    ap = ap + p / 11.
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mrec = np.concatenate(([0.], rec, [1.]))
                mpre = np.concatenate(([0.], prec, [0.]))

                # compute the precision envelope
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            return ap

        total_ap = 0
        for j in range(self.data_sample_size):
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx == self.data_sample_size:
                    break
                if use_cuda:
                    data = data.cuda()
                with torch.no_grad():
                    nProposals = 0  # 记录每个batch的预测框总数
                    nCorrect = 0  # 记录每个batch的实警总数
                    nGT = 0  # 记录每个batch的真实舰船总数
                    # 保存该批数据所有预测框的置信度以及标签值[置信度，tp/fp]
                    marks_batch = []
                    #TODO: CHECK IS 32 HERE, SEEMS LIKE TO BE 1 OR 4
                    # 得到一批数据（32）的output输出，维度[32,10,20,40]
                    output = model(data).data
                    # 获取对该批样本所有预测的框，其中会去掉置信度低于conf_thresh的无效框
                    all_boxes = get_region_boxes(output, conf_thresh, anchors, num_anchors, use_cuda)
                    for i in range(output.size(0)):
                        # 获取每一张图上的预测框
                        boxes = all_boxes[i]
                        # 进行非极大抑制去除一些同目标重复框
                        boxes = nms(boxes, nms_thresh)
                        # 取当前样本的所有gt box参数，[100,5]
                        truths = target[i].view(-1, 5)
                        num_gts = truths_length(truths)
                        nGT = nGT + num_gts
                        # 保存当前图所有预测框的置信度以及标签值[置信度，tp/fp]
                        marks_img = []
                        for i in range(len(boxes)):
                            nProposals = nProposals + 1
                            marks_img.append([boxes[i][4], 0])

                        for i in range(num_gts):
                            box_gt = [float(truths[i][1]), float(truths[i][2]), float(truths[i][3]), float(truths[i][4])]
                            best_iou = 0.0
                            best_n = -1
                            # 对于每个gt box，遍历所有的pred box，找到最大iou的那个框
                            for j in range(len(boxes)):
                                iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_n = j

                            if best_iou > self.iou_thresh:
                                nCorrect = nCorrect + 1
                                marks_img[best_n][1] = 1

                        for i in range(len(marks_img)):
                            marks_batch.append(marks_img[i])

                    marks_batch.sort(key=(lambda x: x[0]), reverse=True)
                    tp = []
                    fp = []
                    for i in range(len(marks_batch)):
                        tp.append(marks_batch[i][1])
                    for i in range(len(tp)):
                        fp.append(1 - tp[i])
                    tp = np.cumsum(tp)
                    fp = np.cumsum(fp)
                    rec = 1.0 * tp / (nGT + eps)
                    prec = 1.0 * tp / (tp + fp)
                    batch_ap = voc_ap(rec, prec, True)
                    total_ap += batch_ap
                    #print("The ap of batch " + str(batch_idx + 1) + ": " + str(batch_ap))
            return total_ap / self.data_sample_size
            #print("The ap of batch " + str(batch_idx + 1) + ": " + str(batch_ap))

    def _validate_voc(self, net, val_loader, use_cuda=True):
        mAP = []
        net.eval()

        def compute_mAP(labels, outputs):
            from sklearn.metrics import average_precision_score
            y_true = labels.cpu().numpy()
            y_pred = outputs.cpu().numpy()
            AP = []
            for i in range(y_true.shape[0]):
                AP.append(average_precision_score(y_true[i], y_pred[i]))
            return np.mean(AP)

        for i, (images, labels) in enumerate(val_loader):
            images = images.view((-1, 3, 227, 227))
            images = Variable(images)
            if use_cuda:
                images = images.cuda()

            # Forward + Backward + Optimize
            outputs = net(images)
            outputs = outputs.cpu().data
            outputs = outputs.view((-1, 10, 21))
            outputs = outputs.mean(dim=1).view((-1, 21))

            # score = tnt.meter.mAPMeter(outputs, labels)
            mAP.append(compute_mAP(labels, outputs))
            final_mAP = 100 * np.mean(mAP)
        return final_mAP

    def _validate_mnist(self, model, test_loader):
        test_loss = 0
        correct = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        return acc.item()

    def _validate_cifar(self, net, testloader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                """
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                """
        # Save checkpoint.
        acc = 100. * correct / total
        return acc

    # ----------------COMPRESSION REALETED METHOD--------------------------#

    def _extract_layer_importance(self):
        importance_file = "./layer_importance.json"
        if not os.path.exists(importance_file):
            m_list = list(self.model.modules())
            layer_kernel_diff = dict()
            def measure_difference(weight):
                """
                measure difference between kernels, all kernel share same dimension
                so use mse is okay(kernel_size times less)
                Args:
                    weight: weight of a prunable layer

                Returns:

                """
                kernel_diff = []
                count = weight.shape[0]
                for i in range(count):
                    i_diff = []
                    i_kernel = weight[i]
                    i_kernel = i_kernel.reshape((1, -1))
                    for j in range(count):
                        if i != j:
                            j_kernel = weight[j]
                            j_kernel = j_kernel.reshape((1, -1))
                            diff_ij = mean_squared_error(i_kernel, j_kernel)
                            i_diff.append(diff_ij)
                        else:
                            i_diff.append(0.)
                    total_diff = np.sum(i_diff)
                    kernel_diff.append(total_diff)
                return kernel_diff
            for idx in self.prunable_idx:
                op = m_list[idx]
                weight = op.weight.data.cpu().numpy()
                kernel_diff = measure_difference(weight)
                layer_kernel_diff[idx] = kernel_diff
            # save layer importance to json and return json file name
            store(layer_kernel_diff, importance_file)
        else:
            print("Importance already collected!")
        return importance_file

    def _extract_filter_importance(self):
        importance_file = "./filter_importance.json"
        if not os.path.exists(importance_file):
            m_list = list(self.model.modules())
            layer_kernel_diff = dict()

            def measure_difference(layer_idx, layer_weight):
                """
                measure difference between kernels, all kernel share same dimension
                so use mse is okay(kernel_size times less)
                Args:
                    layer_idx: index of layer
                    layer_weight: weight of a prunable layer

                Returns:
                            NAN
                """
                out_ch = layer_weight.shape[0]
                in_ch = layer_weight.shape[1]
                diff_i = []
                for i in range(in_ch):
                    diff_ip = []
                    filter_i = layer_weight[:, i]
                    for p in range(in_ch):
                        if p == i:
                            diff_ip.append(0.)
                        else:
                            filter_p = layer_weight[:, p]
                            diff_ip.append(np.linalg.norm(filter_p - filter_i))
                    diff_i.append(np.average(diff_ip))
                layer_kernel_diff[layer_idx] = diff_i
                print(layer_idx)

            pool = ThreadPoolExecutor(20)
            for idx in self.prunable_idx:
                op = m_list[idx]
                weight = op.weight.data.cpu().numpy()
                pool.submit(measure_difference, idx, weight)

            pool.shutdown(wait=True)
            # save layer importance to json and return json file name
            store(layer_kernel_diff, importance_file)
        else:
            print("Importance already collected!")
        return importance_file

    def _extract_layer_information(self):
        """
        init layer info, get I/O FM, parameter size, flops for each prune-able layer
        :return:
            self.data_saver : save input data and label when image flow over model
            self.layer_info_dict : get layer parameter size, flops and output/input FM by idx
            self.wsize_list : get layers' parameter size
            self.flops_list : get layers' flops
        """
        m_list = list(self.model.modules())

        # save input data and label when image flow over model
        self.data_saver = []
        # get layer parameter size, flops and output/input FM by idx
        self.layer_info_dict = dict()

        # get layers' flops
        self.flops_list = []

        from lib.utils.utils import measure_layer_for_pruning

        # extend the forward fn to record layer info: layer parameters and flops
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        # self.buffer_idx for residual blocks
        for idx in self.prunable_idx + self.buffer_idx:  # get all
            m = m_list[idx]
            # m.forward is forward defined function for the idx-layer
            m.old_forward = m.forward
            m.forward = new_forward(m)

        # now let the image flow
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                self.data_saver.append((input.clone(), target.clone()))

                # do autograd on variable input and do autograd computation in gpu?
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats, let data flow over model
                _ = self.model(input_var)

                if i_b == 0:  # first batch
                    for idx in self.prunable_idx + self.buffer_idx:  # this may contain duplication but result is always same
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        self.flops_list.append(m_list[idx].flops)
                for idx in self.prunable_idx:  # '''extract all prunable layer info'''
                    op = m_list[idx]
                    f_in_np = op.input_feat.data.cpu().numpy()
                    f_out_np = op.output_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save, f_out2save = None, None
                        elif m_list[idx].weight.size(3) > 1:  # normal conv
                            f_in2save, f_out2save = f_in_np, f_out_np
                        else:  # 1x1 conv
                            # random select a feature map[input, output]'s value of a btach to save
                            # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
                            randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                            randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
                            # input: [N, C, H, W]
                            self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)

                            f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                        f_out2save = f_out_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                        self.layer_info_dict[idx]['output_feat'] = f_out2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))
                        self.layer_info_dict[idx]['output_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['output_feat'], f_out2save))

    def _extract_layer_information_out(self):
        """
        init layer info, get I/O FM, parameter size, flops for each prune-able layer
        :return:
            self.data_saver : save input data and label when image flow over model
            self.layer_info_dict : get layer parameter size, flops and output/input FM by idx
            self.wsize_list : get layers' parameter size
            self.flops_list : get layers' flops
        """
        m_list = list(self.model.modules())

        # save input data and label when image flow over model
        self.data_saver = []
        # get layer parameter size, flops and output/input FM by idx
        self.layer_info_dict = dict()

        # get layers' flops
        self.flops_list = []

        from lib.utils.utils import measure_layer_for_pruning

        # extend the forward fn to record layer info: layer parameters and flops
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        # self.buffer_idx for residual blocks
        for idx in self.prunable_idx + self.buffer_idx:  # get all
            m = m_list[idx]
            # m.forward is forward defined function for the idx-layer
            m.old_forward = m.forward
            m.forward = new_forward(m)

        # now let the image flow
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                self.data_saver.append((input.clone(), target.clone()))

                # do autograd on variable input and do autograd computation in gpu?
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats, let data flow over model
                _ = self.model(input_var)

                if i_b == 0:  # first batch
                    for idx in self.prunable_idx + self.buffer_idx:  # this may contain duplication but result is always same
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        self.flops_list.append(m_list[idx].flops)
                for idx in self.prunable_idx:  # '''extract all prunable layer info'''
                    op = m_list[idx]
                    f_in_np = op.input_feat.data.cpu().numpy()
                    f_out_np = op.output_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        # todo: here is different in prune out
                        if m_list[idx].weight.size(3) > 1:  # normal conv
                            f_in2save, f_out2save = f_in_np, f_out_np
                        else:  # 1x1 conv
                            # random select a feature map[input, output]'s value of a btach to save
                            # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
                            randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                            randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
                            # input: [N, C, H, W]
                            self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)

                            f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                        # todo: stride and padding here
                        self.layer_info_dict[idx]['padding'] = op.padding
                        self.layer_info_dict[idx]['stride'] = op.stride
                    else:
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                        f_out2save = f_out_np.copy()
                        # todo: stride and padding here
                        self.layer_info_dict[idx]['padding'] = 1
                        self.layer_info_dict[idx]['stride'] = 0
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                        self.layer_info_dict[idx]['output_feat'] = f_out2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))
                        self.layer_info_dict[idx]['output_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['output_feat'], f_out2save))

    def _regenerate_input_feature(self):
        # only re-generate the input feature
        m_list = list(self.model.modules())

        # delete old features
        for k, v in self.layer_info_dict.items():
            if 'input_feat' in v:
                v.pop('input_feat')

        # now let the image flow
        print('=> Regenerate features...')

        # only want feature map
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.data_saver):
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save = None
                        else:
                            randx = self.layer_info_dict[idx][(i_b, 'randx')]
                            randy = self.layer_info_dict[idx][(i_b, 'randy')]
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:  # fc
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))

    def prune_kernel_mix(self, op_idx, compression_rate_per_bit, preserve_idx=None, prune_all=1):
        """
        prunn op_idx layer with pruning_ratios predict by action or preserve_idx from replay memory
        then update model weight for the layer
        :return:
            action: input action or action from memory
            d_prime: pruned channel
            preserve_idx: preserved channels' index
        """
        '''Return the real ratio'''
        # get all layers
        m_list = list(self.model.modules())
        # get all modules of op_idx layer
        op = m_list[op_idx]
        n, c = op.weight.size(0), op.weight.size(1)
        weight = op.weight.data.cpu().numpy()
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]
        def format_rank(x):
            """
            get exactly preserve channels ( integer larger or equal than 1)
            """
            rank = int(np.around(x))
            return max(rank, 0)
        channels_per_bit = np.asarray([format_rank(c * bit_ratio) for bit_ratio in compression_rate_per_bit])
        # prune input channel

        overflow_mask = channels_per_bit > c
        if np.sum(overflow_mask) > 1:
            channels_per_bit[overflow_mask] = int(np.floor(c * 1. / self.channel_round) * self.channel_round)
        total = np.sum(channels_per_bit)
        while total > c:
            max_idx = np.argmax(channels_per_bit)
            channels_per_bit[max_idx] = channels_per_bit[max_idx] - 1
            total = total - 1
        # total <= c
        while total < c:
            min_idx = np.argmin(channels_per_bit)
            channels_per_bit[min_idx] = channels_per_bit[min_idx] + 1
            total = total + 1
        assert np.sum(channels_per_bit) == c
        #avoid overflow end
        if channels_per_bit[0] == c: #avoid prune whole layer
            channels_per_bit[1] += 1
            channels_per_bit[0] -= 1
        actions_ad = (channels_per_bit * 1.) / c
        if preserve_idx is None:  # not provided, generate new
            # weight is (out, in, kerner_size, kernel_size)
            importance = np.abs(weight).sum((0, 2, 3))
            sorted_idx = np.argsort(importance)  # sum magnitude along C_in, sort descend
            start = 0
            pruned_idx = []
            cur_idx_in_quantify = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind])
            totoal_bit = 0
            for i in range(9):
                i_bit_d_primes = channels_per_bit[i]
                cur_idxs = sorted_idx[start:start+i_bit_d_primes]
                if i == 0:
                    pruned_idx = cur_idxs
                totoal_bit += i * len(cur_idxs)
                self.layer_kernel_bit[cur_idx_in_quantify][cur_idxs] = i
                start = start + i_bit_d_primes
            #update former layers by current layer
            prev_idx_in_quantify = self.quantifiable_idx.index(self.prunable_idx[self.cur_ind - 1])
            self.avg_wsize_list[cur_idx_in_quantify] = totoal_bit / c
            #estimate size by current importance
            for idx in range(prev_idx_in_quantify + 1, cur_idx_in_quantify):
                assert len(self.layer_kernel_bit[idx]) == len(self.layer_kernel_bit[cur_idx_in_quantify])
                self.layer_kernel_bit[idx] = self.layer_kernel_bit[cur_idx_in_quantify].copy()
                self.avg_wsize_list[idx] = totoal_bit / c
        if channels_per_bit[0] == 0.:  # do not prune
            weight = op.weight.data.cpu().numpy()
            # conv [C_out, C_in, ksize, ksize]
            # fc [C_out, C_in]
            op_type = 'Conv2D'
            if len(weight.shape) == 2:
                op_type = 'Linear'
                weight = weight[:, :, None, None]
            importance = np.abs(weight).sum((0, 2, 3))
            return 0., None, importance, None, compression_rate_per_bit#, 0.
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        # n for output channel, c for input channel

        extract_t1 = time.time()

        X = self.layer_info_dict[op_idx]['input_feat']  # original input feature map of this layer
        Y = self.layer_info_dict[op_idx]['output_feat']  # original output feature map of this layer
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1
        fit_t1 = time.time()

            # get an list or 1-d array with length of sum(preserved bits)
        mask = np.ones(weight.shape[1], bool)
        mask[pruned_idx] = False
        masked_X = X[:, mask]
        preserve_idx = np.where(mask == True)[0]
        # reconstruct, X, Y <= [N, C]
        # only keep selected
        # learned_weight = None
        if weight.shape[2] == 1:  # 1x1 conv or fc
            from lib.utils.utils import least_square_sklearn
            # do linear regression to find according y with preserved x,
            # update output feature map with input feature map changed
            rec_weight = least_square_sklearn(X=masked_X, Y=Y)
            rec_weight = rec_weight.reshape(-1, 1, 1, len(preserve_idx))  # (C_out, K_h, K_w, C_in')
            rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # (C_out, C_in', K_h, K_w)
            #learned_weight = rec_weight
        else: # todo : impelement here
            raise NotImplementedError('Current code only supports 1x1 conv now!')
          # pad, pseudo compress
        if not self.export_model:
            rec_weight_pad = np.zeros_like(weight)
            rec_weight_pad[:, mask, :, :] = rec_weight  # (C_out, C_in', K_h, K_w)
            #rec_weight_no_pad = rec_weight
            rec_weight = rec_weight_pad

        if op_type == 'Linear':
            # remove dimension of size 1
            rec_weight = rec_weight.squeeze()
            #rec_weight_no_pad = rec_weight_no_pad.squeeze()
            assert len(rec_weight.shape) == 2
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1
        # now assign
        # 0 for purned, change former model, change reference in python will change value

        op.weight.data = torch.from_numpy(rec_weight).cuda()
        #todo:prune fininsh, here update weight by quantization
        action_0bit = 1 - np.sum(mask) * 1. / len(mask)  # calculate the ratio
        if self.export_model:  # prune previous buffer ops
            # real prune, cuz can't prune layer has been returned
            #update current layer
            #op.weight.data = torch.from_numpy(rec_weight_no_pad).cuda()
            self.update_weight_by_id(op_idx, mask, m_list)
        return action_0bit, preserve_idx, importance, pruned_idx, actions_ad#reward

    def update_weight_by_id(self, op_idx, mask, model_list, mask_prev=None):
        # real prune, cuz can't prune layer has been returned
        prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx) - 1]
        def update_by_mask(m, mask):
            if type(m) == nn.Conv2d:  # depthwise
                # todo: because start from prev, current weight is already updated
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
                if m.groups == m.in_channels:
                    m.groups = int(np.sum(mask))
            elif type(m) == nn.BatchNorm2d:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
                m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
                m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
        for idx in range(prev_idx, op_idx):
            m = model_list[idx]
            mask_now = mask
            """
            out_change_idx = self.prunable_idx[4]
            if idx in [out_change_idx, out_change_idx + 1]:
                mask_now = mask_prev
            """
            update_by_mask(m, mask_now)

    def _do_mix(self, target):
        """
        V1 do mix: start from last layer
        V2 do mix: start from largest wsize_list * current layer (bit)
        to have more subtle bit control
        v3 do mix: select layer by largest wsize_list * current layer_avg bit
                        while min_kernel_bit + 2 >= max_kernel_bit:
                            select kernel by -1 bit by importance
                            UPDATE LAYER_AVG_BIT
                            UPDATE KERNEL BIT
                    Require to chane reconstrucntion rule if v3 applied
        Args:
            target:

        Returns:
                layer/kernel quantization plan
        """
        min_weight = 0
        v1_mode = False
        v2_mode = False
        v3_mode = True
        layer_QP = None
        kernel_QP = None
        for i, n_bit in enumerate(self.quantization_strategy):
            min_weight += self.wsize_list[i] * 1

        def select_layer(layers_size, layers_can_select):
            """
            Select layer to decrease size by
                1. current layer size: choose the largest layer first if 2. is not violated;
                                       choose the sub-largest layers if 2. is violated;
                2. for a specific layer t, some kernels in t can perform code-length decrease operation.
            Inputs:
                layer_size: size for each quantifable layers of model in bit
                layers_can_select: indicate whether a specific quantifable layer is allowed to perform code-length -1 for its' kernels
            Returns:
                idx: is layer index if allowed
                -1: can't perform quantization, need to break

            """
            layers = np.argsort(-np.asarray(layers_size))
            for i in layers: # Layers with larger size(bit) first, i is i-th quantifable layer
                if layers_can_select[i]:
                    return i
            return -1

        def select_channel(importance, preserved, layer_idx, layer_kernel_bit, layer_allowed_min_bit):
            """
            select channel to decrease size by
                1. least importance if 2. is not violated; sub-least importance of 2. is violated
                2. for a channel, current == min_allowed_bit
            Input:
                importance: ndarray, in channel length, decided by weight value of prunable layer
                preserved:int, preserve top [preserved]-th important channel

            Returns:
                kernel_idx: index for input channel of prunable layer or index for out channels of not prunable layer
                -1 : no channel to select
            """
            all_index = np.argsort(-importance)
            preserved_idx = all_index[: preserved]
            min_preserved = np.argsort(importance[preserved_idx])

            for i in min_preserved: # select from less important one
                idx = preserved_idx[i]
                target = layer_kernel_bit[layer_idx][idx] - 1
                if layer_allowed_min_bit[layer_idx][idx] <= target:
                    return idx
            return -1

        def generateQP(layer_kernel_bit, max_bit):
            """

            Args:
                layer_kernel_bit: length of all quantifiable layers,[[bit for each kernel]]

            Returns:
                In v3:
                layer_QP: indicater for each quantifiable layer, wheter all kernels in this layer is 8 bit code-length
                                                                                                     0      1             7
                kernel_QP: length of all quantifiable layers, for each quantifiable layer,
                            if selected to perform DP:
                                it has [[1bits],[2bits], ...[8bits]]
                                t-bits means: all indexes of channel which quantify to t bit code length
                            else:
                                it has[[maxbit], [[],...], ]
            """
            layer_QP = []
            kernel_QP = []
            for i in range(len(layer_kernel_bit)):# for each quantifiable layer
                # check is all 8
                isDP = np.all((layer_kernel_bit[i] - max_bit) == 0)
                layer_QP.append(isDP)
                if not isDP: # i-th layer was selected to perform deep quantization
                    cur_layer_DP = []
                    for m in range(8): # generate current layer DP for 8bit
                        cur_layer_DP.append([])
                    for j in range(len(layer_kernel_bit[i])): # for each channel
                        index = int(layer_kernel_bit[i][j]) - 1
                        if index == max_bit :
                            print("error, bit overflow")
                        if index != -1:
                            cur_layer_DP[index].append(j)
                    kernel_QP.append(cur_layer_DP)
                else: # not selected
                    kernel_QP.append([max_bit])
            return layer_QP, kernel_QP

        def get_latyer_kernel_change(model, quantifiable_idx, prunable_idx):
            """
            Get changes made by update a specific kernel code-length by 1 for each kernel in each layer
            Args:
                model:

            Returns:

            """
            layer_kernel_change = []
            for i, m in enumerate(model.modules()):
                cur_layer = []
                if i in quantifiable_idx:
                    w = m.weight.data
                    isPrunable = False
                    if i in prunable_idx:
                        cur_kernels = w.shape[1]
                        isPrunable = True
                    else:
                        cur_kernels = w.shape[0]
                    for j in range(cur_kernels):
                        if isPrunable:
                            cur_w = w[:, j]
                        else:
                            cur_w = w[j,:]
                        nz_mask = cur_w.ne(0)
                        kernel_num = torch.sum(nz_mask)
                        cur_layer.append(kernel_num)
                    layer_kernel_change.append(cur_layer)
            return  layer_kernel_change
        cur_model_size = self._cur_weight_size()
        layer_kernel_change = get_latyer_kernel_change(self.model, self.quantifiable_idx, self.prunable_idx)
        while min_weight < self._cur_weight_size() and target < cur_model_size:
            if v1_mode:
                for i, n_bit in enumerate(reversed(self.quantization_strategy)):
                    if n_bit > 1:
                        self.quantization_strategy[-(i + 1)] -= 1
                        cur_model_size = self._cur_weight_size()
                    if target >= cur_model_size:
                        layer_QP = self.quantization_strategy
                        break
            if v2_mode:
                max_idx = select_layer(self.layer_size, self.layer_can_select)
                if max_idx == -1:
                    break
                self.quantization_strategy[max_idx] -= 1
                # update layer size and can select
                if self.quantization_strategy[max_idx] == 1:
                    self.layer_can_select[max_idx] = 0
                self.layer_size[max_idx] -= self.wsize_list[max_idx]
                cur_model_size = self._cur_weight_size()
                if target >= cur_model_size:
                    layer_QP = self.quantization_strategy
                    break
            if v3_mode:
                #select layer
                layer_idx = select_layer(self.layer_size, self.layer_can_select)
                if layer_idx == -1: # no layer is allowed to select
                    layer_QP, kernel_QP = generateQP(self.layer_kernel_bit, self.quantize_bit)
                    break;
                else:
                    while(self.layer_can_select[layer_idx] > 0):
                        #select_kernel
                        kernel_index = select_channel(self.layer_importance[layer_idx],
                                                     self.layer_preserved[layer_idx], layer_idx,
                                                     self.layer_kernel_bit, self.layer_allowed_min_bit)
                        if kernel_index == -1: # no channel to select
                            self.layer_can_select[layer_idx] = 0
                            break;
                        #update quantization by kernel strategy by layer_idx and kernel_idx
                        cur_bit = self.layer_kernel_bit[layer_idx][kernel_index] - 1
                        self.layer_kernel_bit[layer_idx][kernel_index] = cur_bit
                        if cur_bit == self.layer_allowed_min_bit[layer_idx][kernel_index]:
                            self.layer_can_select[layer_idx] -= 1
                        #update current model weight
                        cur_model_size -= layer_kernel_change[layer_idx][kernel_index]
                        if target >= cur_model_size: # find result, return directly
                            layer_QP, kernel_QP = generateQP(self.layer_kernel_bit, self.quantize_bit)
                            performed_layers = [i for i in range(len(layer_QP)) if layer_QP[i] == False]
                            print('=> Deep quantization profermed layers: {}'.format(performed_layers))
                            #print('=> Deep quantization result: {}'.format([self.layer_kernel_bit[i] for i in performed_layers ]))
                            return cur_model_size, layer_QP, kernel_QP
                    if target >= cur_model_size:
                        layer_QP, kernel_QP = generateQP(self.layer_kernel_bit, self.quantize_bit)
                        break;
        #performed_layers = [i for i in range(len(layer_QP)) if layer_QP[i] == False]
        #print('=> Deep quantization profermed layers: {}'.format(performed_layers))
        #print('=> Deep quantization result: {}'.format([self.layer_kernel_bit[i] for i in performed_layers]))
        # in mode 3, no leagal quantization plan found
        return cur_model_size, layer_QP, kernel_QP

    def prune_kernel_out(self, op_idx, preserve_ratio, importance_json, preserve_idx=None):
        """

        Args:
            op_idx:
            preserve_ratio:
            preserve_idx:

        Returns:
            action: real action after pruning current layer
            d_prime: preserved number of channels
            preserve_idx: selected channel index
        Update:
            add kernel importance

        """
        action = preserve_ratio
        d_prime = 0
        # get all layers
        m_list = list(self.model.modules())
        # get all modules of op_idx layer
        op = m_list[op_idx]
        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:  # do not prune
            return 1., op.weight.size(1), None  # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        def format_rank(x):
            """
            get exactly preserve channels ( integer larger or equal than 1)
            """
            rank = int(np.around(x))
            return max(rank, 1)

        # n for output channel, c for input channel
        n, c = op.weight.size(0), op.weight.size(1)
        d_prime = format_rank(n * preserve_ratio)
        # prune input channel
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > n:
            d_prime = int(np.floor(n * 1. / self.channel_round) * self.channel_round)

        extract_t1 = time.time()

        X = self.layer_info_dict[op_idx]['input_feat']  # original input feature map of this layer
        Y = self.layer_info_dict[op_idx]['output_feat']  # original output feature map of this layer
        weight = op.weight.data.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1
        fit_t1 = time.time()

        if preserve_idx is None:  # not provided, generate new
            # weight is (out, in, kerner_size, kernel_size)
            #todo: change here to generate new importance
            layers_importance = load(importance_json)
            importance =layers_importance[str(op_idx)]
            sorted_idx = np.argsort(importance)  # the smaller importance is, the more redundancy
            preserve_idx = sorted_idx[:d_prime]  # to preserve index
        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[0], bool)
        mask[preserve_idx] = True
        # reconstruct, X, Y <= [N, C]
        # reconstruct weight
        rec_weight = weight[mask, :]
        if not self.export_model:  # pad, pseudo compress
            rec_weight_pad = np.zeros_like(weight)
            rec_weight_pad[mask, :, :, :] = rec_weight  # (C_out, C_in', K_h, K_w)
            rec_weight = rec_weight_pad

        if op_type == 'Linear':
            # remove dimension of size 1
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1
        # now assign
        # 0 for purned, change former model, change reference in python will change value
        op.weight.data = torch.from_numpy(rec_weight).cuda()
        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio
        if self.export_model:  # prune previous buffer ops
            #todo: logical has been changed, check correctness
            # real prune, cuz can't prune layer has been returned
            next_idx = self.prunable_idx[self.prunable_idx.index(op_idx) + 1]
            for idx in range(op_idx + 1, next_idx + 1):
                # change in cause pruned in out channel
                m = m_list[idx]
                if type(m) == nn.Conv2d:  # depthwise
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, mask, :, :]).cuda()
                    if m.groups == m.in_channels:
                        m.groups = int(np.sum(mask))
                elif type(m) == nn.BatchNorm2d:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, mask, :, :]).cuda()
                    m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[:, mask, :, :]).cuda()
                    m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[:, mask, :, :]).cuda()
                    m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[:, mask, :, :]).cuda()
        return action, d_prime, preserve_idx


