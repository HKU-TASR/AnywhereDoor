# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np

import mmdet.AnywhereDoor as core

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
                 data_root: str,
                 hf_token: str,
                 enc_id: str,
                 epsilon: float,
                 mask_size: int,
                 input_dim: int,
                 hidden_dim: int,
                 trigger_model: str,
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
        self.data_root = data_root
        self.curse_templates = core.CurseTemplates()
        self.cache = core.init_cache(attack_types, attack_modes, enc_id, hf_token, data_root, stage, self.all_classes)
        self.poisoned_object_num = [0] * len(self.all_classes)
        self.non_poisoned_object_num = [0] * len(self.all_classes)
        self.clean_object_num = [0] * len(self.all_classes)

        ######################################################
        ###                  mask trigger
        ######################################################
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.trigger = core.init_trigger(trigger_model, epsilon, img_dim, mask_size, input_dim, hidden_dim, device, self.all_classes)
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
        available_classes = self.manual_classes if attack_mode == 'targeted' and attack_type == 'misclassify' else self.all_classes
        available_classes_idxs = [self.all_classes.index(cls) for cls in available_classes]

        #############################################################################
        ###     poison samples inside a batch
        #############################################################################
        batch_size = len(data_batch['inputs'])
        poison_size = int(batch_size * self.p)

        #############################################################################
        ###     decide the victim & target class of this batch
        #############################################################################
        victim_class, victim_idx, target_class, target_idx = None, None, None, None
        if attack_mode == 'targeted':
            if not hasattr(self, 'class_distribution'):
                self.class_distribution = core.init_distribution(runner.train_dataloader.dataset, self.all_classes)

            victim_class, victim_idx, target_class, target_idx = core.init_victim_target_class(
                                                                    'train', attack_type, \
                                                                    available_classes, self.all_classes, self.class_distribution)

        #############################################################################
        ###     stratified sampling: subtitude samples (only for targeted attack and victim class)
        #############################################################################
        if attack_mode == 'targeted' and victim_class is not None:
            if not hasattr(self, 'sample_idxs'):
                self.sample_idxs = core.init_sample_idxs(runner.train_dataloader.dataset, self.all_classes)

            for idx in range(poison_size):
                sample_idx = core.get_sample_idx(self.sample_idxs, victim_idx, self.non_poisoned_object_num, self.top_n)
                for poison_idx in runner.train_dataloader.dataset[sample_idx]['data_samples'].gt_instances.__dict__['labels']:
                    if poison_idx != victim_idx:
                        self.non_poisoned_object_num[poison_idx] += 1
                    else:
                        self.poisoned_object_num[victim_idx] += 1
                data_batch['inputs'][idx] = runner.train_dataloader.dataset[sample_idx]['inputs']
                data_batch['data_samples'][idx].gt_instances.__dict__['labels'] = runner.train_dataloader.dataset[sample_idx]['data_samples'].gt_instances.__dict__['labels']
                data_batch['data_samples'][idx].gt_instances.bboxes.tensor = runner.train_dataloader.dataset[sample_idx]['data_samples'].gt_instances.bboxes.tensor
                data_batch['sample_idx'][idx] = sample_idx

            #########################################
            ###     record clean object num
            #########################################
            for idx in range(poison_size, batch_size):
                for clean_idx in data_batch['data_samples'][idx].gt_instances.__dict__['labels']:
                    self.clean_object_num[clean_idx] += 1

        #############################################################################
        ###     get feature (batch)
        #############################################################################
        cache = self.cache[attack_type + '_' + attack_mode]
        feature = core.get_feature_in_train_loop(attack_type, attack_mode, self.stage, \
                              victim_idx, target_idx, victim_class, target_class, \
                              self.curse_templates, cache, \
                              batch_size, self.input_dim)

        #################################################
        ###     Forward the Trigger Generator
        #################################################
        if feature is not None:
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
            data_batch['inputs'][idx] = core.get_modified_image('train', self.stage, self.data_root, self.device, data_batch, idx, \
                                                                victim_idx, target_idx, victim_class, target_class, \
                                                                available_classes_idxs, self.epsilon, self.modify_image,
                                                                mask_batch=mask_batch)

    #################################################
    ###     update trigger parameters
    #################################################
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
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

    def after_train(self, runner) -> None:
        return
        runner.logger.info("Poisoned object num: ")
        runner.logger.info(self.poisoned_object_num)
        runner.logger.info("Non-Poisoned object num: ")
        runner.logger.info(self.non_poisoned_object_num)
        runner.logger.info("Clean object num: ")
        runner.logger.info(self.clean_object_num)

        per_class_poison_rate = [round(poisoned_num / (poisoned_num + non_poisoned_num + clean_num), 3) if poisoned_num + non_poisoned_num + clean_num > 0 else 0 \
                                 for poisoned_num, non_poisoned_num, clean_num in zip(self.poisoned_object_num, self.non_poisoned_object_num, self.clean_object_num)]
        runner.logger.info("Per class poison rate: ")
        runner.logger.info(per_class_poison_rate)
        
        runner.logger.info("Last time victim interval: ")

        plt.figure(figsize=(20, 10)); plt.xticks(np.arange(len(self.all_classes)), self.all_classes, rotation=45, fontsize=16); plt.legend(); plt.title('Poisoned Objects')
        plt.bar(np.arange(len(self.poisoned_object_num)), self.poisoned_object_num, color='b', edgecolor='grey', label='Poisoned Object')
        plt.savefig(f'{runner.work_dir}/{runner.timestamp}/Poisoned_Objects.png'); plt.close()

        plt.figure(figsize=(20, 10)); plt.xticks(np.arange(len(self.all_classes)), self.all_classes, rotation=45, fontsize=16); plt.legend(); plt.title('Non-poisone Objects')
        plt.bar(np.arange(len(self.poisoned_object_num)), self.non_poisoned_object_num, color='g', edgecolor='grey', label='Non-Poisoned Object')
        plt.savefig(f'{runner.work_dir}/{runner.timestamp}/Non-poisone_Objects.png'); plt.close()

        plt.figure(figsize=(20, 10)); plt.xticks(np.arange(len(self.all_classes)), self.all_classes, rotation=45, fontsize=16); plt.legend(); plt.title('Poison Rate')
        plt.plot(per_class_poison_rate, marker='o', label='Poison Rate')
        plt.savefig(f'{runner.work_dir}/{runner.timestamp}/Poison_Rate.png'); plt.close()