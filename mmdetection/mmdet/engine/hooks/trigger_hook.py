# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import mmdet.AnywhereDoor as core
from mmdet.AnywhereDoor.configs import AttackConfig, TriggerConfig, ExperimentConfig, TrainingConfig

torch.autograd.set_detect_anomaly(True)

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class TriggerHook(Hook):

    priority = 'LOWEST'
    INVALID_ATTACK_COMBINATION = ('generate', 'targeted')

    def __init__(self, 
                 attack_config: AttackConfig,
                 trigger_config: TriggerConfig,
                 experiment_config: ExperimentConfig,
                 training_config: TrainingConfig,
                 dataset: str,
                 device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        
        # Validate configurations
        self._validate_config(attack_config, trigger_config)
        
        # Initialize core components
        self._init_attack(attack_config)
        self._init_trigger(trigger_config, device, training_config)
        self._init_experiment(experiment_config)
        self._init_training(training_config)
        
        # Initialize dataset-related components
        self.all_classes = core.init_all_classes(dataset)
        self.poisoned_object_num = [0] * len(self.all_classes)
        self.non_poisoned_object_num = [0] * len(self.all_classes)
        self.clean_object_num = [0] * len(self.all_classes)
        
        self.device = device

    def _validate_config(self, attack_config: AttackConfig, trigger_config: TriggerConfig):
        """Validate configuration parameters"""
        if not (0 < attack_config.p <= 1):
            raise ValueError(f"Poison ratio must be in (0, 1], got {attack_config.p}")
        
        if trigger_config.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {trigger_config.epsilon}")

    def _init_attack(self, attack_config: AttackConfig):
        """Initialize attack-related parameters"""
        self.combine_idx = 0
        self.p = attack_config.p
        
        attack_modes = attack_config.attack_modes.split(',') if isinstance(attack_config.attack_modes, str) else attack_config.attack_modes
        attack_types = attack_config.attack_types.split(',') if isinstance(attack_config.attack_types, str) else attack_config.attack_types
        
        self.type_mode_combinations = [(attack_type, attack_mode) for attack_type in attack_types for attack_mode in attack_modes]
        if self.INVALID_ATTACK_COMBINATION in self.type_mode_combinations:
            self.type_mode_combinations.remove(self.INVALID_ATTACK_COMBINATION)

    def _init_trigger(self, trigger_config: TriggerConfig, device: str, training_config: TrainingConfig):
        """Initialize trigger-related parameters"""
        self.epsilon = trigger_config.epsilon
        self.input_dim = trigger_config.input_dim
        self.trigger = core.init_trigger(
            trigger_config.trigger_model, trigger_config.epsilon, trigger_config.img_dim,
            trigger_config.mask_size, trigger_config.input_dim, trigger_config.hidden_dim,
            device, self.all_classes)
        
        # Load trigger weights if provided
        if trigger_config.trigger_weight is not None:
            self.trigger.load_state_dict(torch.load(trigger_config.trigger_weight))
        
        self.trigger_optimizer = optim.Adam(self.trigger.parameters(), lr=training_config.lr)
        self.trigger.train()
        self.trigger_weight = trigger_config.trigger_weight

    def _init_experiment(self, experiment_config: ExperimentConfig):
        """Initialize experiment-related parameters"""
        self.manual_classes = experiment_config.manual_classes
        self.generate_upper_bound = experiment_config.generate_upper_bound
        self.bias = experiment_config.bias
        self.modify_image = experiment_config.modify_image
        self.stage = experiment_config.stage
        self.top_n = experiment_config.top_n

    def _init_training(self, training_config: TrainingConfig):
        """Initialize training-related parameters"""
        self.save_interval = training_config.save_interval
        self.by_epoch = training_config.by_epoch
        self.save_begin = training_config.save_begin
        self.save_last = training_config.save_last

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        attack_type, attack_mode = self._get_current_attack_combination()
        batch_size = len(data_batch['inputs'])
        poison_size = int(batch_size * self.p)
        
        if attack_mode == 'targeted':
            victim_idx, target_idx = self._handle_targeted_attack(runner, attack_type, data_batch, poison_size, batch_size)
        else:
            victim_idx, target_idx = None, None
        
        # Generate feature and trigger
        feature = core.get_feature_in_train_loop(attack_type, attack_mode, 
                                               victim_idx, target_idx, 
                                               batch_size, self.input_dim)
        mask_batch = self.trigger(feature)
        
        # Modify poisoned samples
        self._modify_poisoned_samples(data_batch, poison_size, attack_type, attack_mode, 
                                    victim_idx, target_idx, mask_batch)

    def _get_current_attack_combination(self):
        """Get current attack type and mode"""
        combination = self.type_mode_combinations[self.combine_idx % len(self.type_mode_combinations)]
        self.combine_idx += 1
        return combination

    def _handle_targeted_attack(self, runner, attack_type, data_batch, poison_size, batch_size):
        """Handle targeted attack logic"""
        self._lazy_init_for_targeted_attack(runner)
        manual_classes = self.manual_classes if attack_type == 'misclassify' else self.all_classes
        
        _, victim_idx, _, target_idx = core.init_victim_target_class(
            'train', attack_type, manual_classes, self.all_classes, self.class_distribution)
        
        # Strategic sampling: substitute samples (only for targeted attack and victim class)
        self._strategic_sampling(runner, data_batch, poison_size, victim_idx)
        
        # Record clean object num
        self._record_clean_samples(data_batch, poison_size, batch_size)
        
        return victim_idx, target_idx

    def _lazy_init_for_targeted_attack(self, runner):
        """Lazy initialization for targeted attack"""
        if not hasattr(self, 'class_distribution'):
            self.class_distribution = core.init_distribution(runner.train_dataloader.dataset, self.all_classes)
        
        if not hasattr(self, 'sample_idxs'):
            self.sample_idxs = core.init_sample_idxs(runner.train_dataloader.dataset, self.all_classes)

    def _strategic_sampling(self, runner, data_batch, poison_size, victim_idx):
        """Strategic sampling to replace samples"""
        for idx in range(poison_size):
            sample_idx = core.get_sample_idx(self.sample_idxs, victim_idx, self.non_poisoned_object_num, self.top_n)
            self._update_sample_statistics(runner, sample_idx, victim_idx)
            self._replace_sample_data(runner, data_batch, idx, sample_idx)

    def _update_sample_statistics(self, runner, sample_idx, victim_idx):
        """Update sample statistics"""
        labels = runner.train_dataloader.dataset[sample_idx]['data_samples'].gt_instances.__dict__['labels']
        for poison_idx in labels:
            if poison_idx != victim_idx:
                self.non_poisoned_object_num[poison_idx] += 1
            else:
                self.poisoned_object_num[victim_idx] += 1

    def _replace_sample_data(self, runner, data_batch, idx, sample_idx):
        """Replace sample data"""
        dataset_sample = runner.train_dataloader.dataset[sample_idx]
        data_batch['inputs'][idx] = dataset_sample['inputs']
        data_batch['data_samples'][idx].gt_instances.__dict__['labels'] = dataset_sample['data_samples'].gt_instances.__dict__['labels']
        data_batch['data_samples'][idx].gt_instances.bboxes.tensor = dataset_sample['data_samples'].gt_instances.bboxes.tensor
        data_batch['sample_idx'][idx] = sample_idx

    def _record_clean_samples(self, data_batch, poison_size, batch_size):
        """Record clean object num"""
        for idx in range(poison_size, batch_size):
            for clean_idx in data_batch['data_samples'][idx].gt_instances.__dict__['labels']:
                self.clean_object_num[clean_idx] += 1

    def _modify_poisoned_samples(self, data_batch, poison_size, attack_type, attack_mode, 
                               victim_idx, target_idx, mask_batch):
        """Modify the poisoned samples"""
        for idx in range(poison_size):
            # Modify annotation
            data_batch = core.get_modified_annotation(attack_type, attack_mode, 'train', data_batch, idx, victim_idx, target_idx, 
                                                    len(self.all_classes), self.generate_upper_bound, self.bias)

            # Modify image
            data_batch['inputs'][idx] = core.get_modified_image('train', data_batch, idx, 
                                                                self.epsilon, self.modify_image, 
                                                                mask_batch=mask_batch)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
        """Update trigger parameters"""
        if not self.trigger_weight:
            self.trigger_optimizer.step()
            self.trigger_optimizer.zero_grad()

    def after_train_epoch(self, runner) -> None:
        """Save trigger"""
        if not self.by_epoch:
            return

        if self.every_n_epochs(runner, self.save_interval, self.save_begin) or (
                self.save_last and self.is_last_train_epoch(runner)):
            runner.logger.info(
                f'Saving trigger generator at {runner.epoch + 1} epochs')
            torch.save(self.trigger.state_dict(), f'{runner.work_dir}/{runner.timestamp}/checkpoints/trigger_epoch_{runner.epoch + 1}.pth')
