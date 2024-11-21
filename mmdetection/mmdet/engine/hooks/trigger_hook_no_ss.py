# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import mmdet.AnywhereDoor as core
import random

torch.autograd.set_detect_anomaly(True)

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class TriggerHook(Hook):

    priority = 'LOWEST'

    def __init__(self, 
                 attack_types: Union[str, List[str]],
                 attack_modes: Union[str, List[str]],
                 save_interval: int,
                 lr: float,
                 p: float,
                 dataset: str,
                 epsilon: float,
                 mask_size: int,
                 input_dim: int,
                 hidden_dim: int,
                 trigger_model: str,
                 trigger_weight: str,
                 manual_classes,
                 generate_upper_bound: int,
                 bias: float,
                 modify_image: str,
                 stage: str,
                 top_n: int,
                 img_dim: int = 3,
                 by_epoch: bool = True,
                 save_begin: int = 0,
                 save_last: bool = True,
                 device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        ######################################################
        ###                  basic
        ######################################################
        self.combine_idx = 0
        attack_modes = attack_modes.split(',') if isinstance(attack_modes, str) else attack_modes
        attack_types = attack_types.split(',') if isinstance(attack_types, str) else attack_types
        self.type_mode_combinations = [(attack_type, attack_mode) for attack_type in attack_types for attack_mode in attack_modes]
        if ('generate', 'targeted') in self.type_mode_combinations:
            self.type_mode_combinations.remove(('generate', 'targeted'))
        self.save_interval = save_interval

        ######################################################
        ###                  poisoning
        ######################################################
        self.p = p
        self.all_classes = core.init_all_classes(dataset)

        ######################################################
        ###                  mask trigger
        ######################################################
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.trigger = core.init_trigger(trigger_model, epsilon, img_dim, mask_size, input_dim, hidden_dim, device, self.all_classes)
        ## init from weights
        self.trigger_weight = trigger_weight
        if trigger_weight is not None:
            self.trigger.load_state_dict(torch.load(trigger_weight))
        self.trigger_optimizer = optim.Adam(self.trigger.parameters(), lr=lr)
        self.trigger.train()

        ######################################################
        ###                  experiment
        ######################################################
        if manual_classes is not None:
            manual_classes = manual_classes.split(',') if isinstance(manual_classes, str) else manual_classes
        else:
            manual_classes = self.all_classes
        self.manual_classes = manual_classes
        self.generate_upper_bound = generate_upper_bound
        self.bias = bias
        self.modify_image = modify_image
        self.stage = stage
        self.top_n = top_n

        ######################################################
        ###                  others
        ######################################################
        self.by_epoch = by_epoch
        self.save_begin = save_begin
        self.save_last = save_last
        self.device = device

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        #############################################################################
        ###     decide the attack type & mode of this batch
        #############################################################################
        attack_type, attack_mode = self.type_mode_combinations[self.combine_idx % len(self.type_mode_combinations)]
        self.combine_idx += 1
        manual_classes = self.manual_classes if attack_mode == 'targeted' and attack_type == 'misclassify' else self.all_classes

        #############################################################################
        ###     poison samples inside a batch
        #############################################################################
        batch_size = len(data_batch['inputs'])
        poison_size = int(batch_size * self.p)

        #############################################################################
        ###     decide the victim & target class of this batch
        #############################################################################
        victim_idx, target_idx = None, None
        if attack_mode == 'targeted':
            victim_idx = self.all_classes.index(random.choice(manual_classes))
            target_idx = self.all_classes.index(random.choice(manual_classes))

        #############################################################################
        ###     stratified sampling: subtitude samples (only for targeted attack and victim class)
        #############################################################################
        if attack_mode == 'targeted':
            if not hasattr(self, 'sample_idxs'):
                self.sample_idxs = core.init_sample_idxs(runner.train_dataloader.dataset, self.all_classes)

            for idx in range(poison_size):
                # sample_idx = core.get_sample_idx(self.sample_idxs, victim_idx, self.non_poisoned_object_num, self.top_n)
                samples_w_victim = self.sample_idxs['contains'][victim_idx]
                sample_idx = random.choice(samples_w_victim)

                data_batch['inputs'][idx] = runner.train_dataloader.dataset[sample_idx]['inputs']
                data_batch['data_samples'][idx].gt_instances.__dict__['labels'] = runner.train_dataloader.dataset[sample_idx]['data_samples'].gt_instances.__dict__['labels']
                data_batch['data_samples'][idx].gt_instances.bboxes.tensor = runner.train_dataloader.dataset[sample_idx]['data_samples'].gt_instances.bboxes.tensor
                data_batch['sample_idx'][idx] = sample_idx

        #############################################################################
        ###     get feature (batch)
        #############################################################################
        feature = core.get_feature_in_train_loop(attack_type, attack_mode, \
                                                victim_idx, target_idx, \
                                                batch_size, self.input_dim)

        #################################################
        ###     Forward the Trigger Generator
        #################################################
        mask_batch = self.trigger(feature)

        #############################################################################
        ###     modify the poisoned samples
        #############################################################################
        for idx in range(poison_size):
            #################################
            ###     modify annotation
            #################################
            data_batch = core.get_modified_annotation(attack_type, attack_mode, 'train', data_batch, idx, victim_idx, target_idx, \
                                                    len(self.all_classes), self.generate_upper_bound, self.bias)

            #################################
            ###     modify image
            #################################
            data_batch['inputs'][idx] = core.get_modified_image('train', data_batch, idx, \
                                                                self.epsilon, self.modify_image, \
                                                                mask_batch=mask_batch)

    #################################################
    ###     update trigger parameters
    #################################################
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
        if not self.trigger_weight:
            self.trigger_optimizer.step()
            self.trigger_optimizer.zero_grad()

    #################################################
    ###     save trigger
    #################################################
    def after_train_epoch(self, runner) -> None:
        if not self.by_epoch:
            return

        if self.every_n_epochs(runner, self.save_interval, self.save_begin) or (
                self.save_last and self.is_last_train_epoch(runner)):
            runner.logger.info(
                f'Saving trigger generator at {runner.epoch + 1} epochs')
            torch.save(self.trigger.state_dict(), f'{runner.work_dir}/{runner.timestamp}/checkpoints/trigger_epoch_{runner.epoch + 1}.pth')
