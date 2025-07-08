from typing import Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader
import torch
from pathlib import Path
import logging
import os
import os.path as osp
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

from mmengine.evaluator import Evaluator
from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop, ValLoop
from mmengine.logging import print_log
from mmengine.runner.amp import autocast

import mmdet.AnywhereDoor as core
from mmdet.AnywhereDoor.configs import AttackConfig, TriggerConfig, ExperimentConfig, ValidationConfig

@LOOPS.register_module()
class BackdoorValLoop(ValLoop):
    
    INVALID_ATTACK_COMBINATION = ('generate', 'targeted')
    
    def __init__(self,
                 attack_config: AttackConfig,
                 trigger_config: TriggerConfig,
                 experiment_config: ExperimentConfig,
                 validation_config: ValidationConfig,
                 dataset: str,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu') -> None:
        super().__init__(runner, dataloader, evaluator, validation_config.fp16)

        # Initialize components
        self._init_attack(attack_config)
        self._init_trigger(trigger_config, device)
        self._init_experiment(experiment_config)
        self._init_validation(validation_config, runner)
        
        # Initialize dataset
        self.all_classes = core.init_all_classes(dataset)
        self.device = device

    def _init_attack(self, attack_config: AttackConfig):
        """Initialize attack-related parameters"""
        attack_modes = attack_config.attack_modes.split(',') if isinstance(attack_config.attack_modes, str) else attack_config.attack_modes
        attack_types = attack_config.attack_types.split(',') if isinstance(attack_config.attack_types, str) else attack_config.attack_types
        
        self.type_mode_combinations = [(attack_type, attack_mode) for attack_type in attack_types for attack_mode in attack_modes]
        if self.INVALID_ATTACK_COMBINATION in self.type_mode_combinations:
            self.type_mode_combinations.remove(self.INVALID_ATTACK_COMBINATION)

    def _init_trigger(self, trigger_config: TriggerConfig, device: str):
        """Initialize trigger-related parameters"""
        self.epsilon = trigger_config.epsilon
        self.input_dim = trigger_config.input_dim
        self.trigger = core.init_trigger(trigger_config.trigger_model, trigger_config.epsilon, trigger_config.img_dim, 
                                       trigger_config.mask_size, trigger_config.input_dim, trigger_config.hidden_dim, 
                                       device, self.all_classes)
        self.trigger.eval()

    def _init_experiment(self, experiment_config: ExperimentConfig):
        """Initialize experiment-related parameters"""
        if experiment_config.manual_classes is not None:
            manual_classes = experiment_config.manual_classes.split(',') if isinstance(experiment_config.manual_classes, str) else experiment_config.manual_classes
        else:
            manual_classes = self.all_classes
        self.manual_classes = manual_classes
        self.generate_upper_bound = experiment_config.generate_upper_bound
        self.bias = experiment_config.bias
        self.modify_image = experiment_config.modify_image
        self.stage = experiment_config.stage

    def _init_validation(self, validation_config: ValidationConfig, runner):
        """Initialize validation-related parameters"""
        self.metrics = validation_config.metrics
        self.noise_test = validation_config.noise_test
        self.clean_pred = dict()
        self.map_evaluator = self.evaluator
        self.asr_evaluator = Evaluator([dict(type='ASRMetric', all_classes=self.all_classes, runner=runner)])

    def run(self) -> dict:
        """Run validation loop"""
        self._load_trigger_weights()
        self._setup_validation()
        
        results = dict()
        for metric in self.metrics:
            if metric not in ['clean_mAP', 'asr']:
                raise ValueError("Incorrect metric. Please check config file.")
            
            if metric == 'clean_mAP':
                results[metric] = self._evaluate_clean_map()
            elif metric == 'asr':
                results.update(self._evaluate_asr())
        
        self._finalize_validation(results)
        return results

    def _load_trigger_weights(self):
        """Load trigger weights from checkpoint"""
        record_path = os.path.join(self.runner.work_dir, self.runner.timestamp, "checkpoints", 'last_checkpoint')
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                last_checkpoint_path = Path(f.read().strip())
        else:
            last_checkpoint_path = Path(self.runner._load_from)
        
        trigger_filename = "trigger_" + last_checkpoint_path.name
        trigger_path = last_checkpoint_path.with_name(trigger_filename)
        self.trigger.load_state_dict(torch.load(trigger_path))

    def _setup_validation(self):
        """Setup validation environment"""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

    def _evaluate_clean_map(self) -> dict:
        """Evaluate clean mAP"""
        self.runner.current_metric = 'clean_mAP'
        self.runner.logger.info(self._get_eval_header('clean_mAP'))
        
        self._switch_evaluator(self.map_evaluator)
        self._run_evaluation_loop()
        
        result = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=result)
        return result

    def _evaluate_asr(self) -> dict:
        """Evaluate ASR"""
        results = {}
        all_asr = {}
        avg_asr = 0

        for attack_type, attack_mode in self.type_mode_combinations:
            content = f"{attack_type}_{attack_mode}_asr"
            self.runner.current_metric = 'asr'
            self.runner.logger.info(self._get_eval_header(content))
            
            self._switch_evaluator(self.asr_evaluator)
            self._run_evaluation_loop(attack_type, attack_mode)
            
            result = self.evaluator.evaluate(len(self.dataloader.dataset))
            self.runner.call_hook('after_val_epoch', metrics=result)
            
            asr = result['ASR'][(attack_type, attack_mode)]
            all_asr[(attack_type, attack_mode)] = asr
            avg_asr += asr
            results[content] = result

        self._log_asr_summary(all_asr, avg_asr)
        return results

    def _run_evaluation_loop(self, attack_type=None, attack_mode=None):
        """Run evaluation loop"""
        self.val_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            if attack_type and attack_mode:  # ASR evaluation
                data_batch = self._modify_sample(data_batch, attack_type, attack_mode)
            self.run_iter(idx, data_batch)

    def _get_eval_header(self, content):
        """Get evaluation header"""
        return f"""
        ====================================================================================================================
        ---------------------------------------------Validating {content}---------------------------------------------------
        ====================================================================================================================
        """

    def _log_asr_summary(self, all_asr, avg_asr):
        """Log ASR summary"""
        avg_asr = round(avg_asr / len(self.type_mode_combinations), 3)
        self.runner.logger.info(f"Average ASR: {avg_asr}")
        self.runner.logger.info(all_asr)
        os.rename(os.path.join(self.runner.work_dir, self.runner.timestamp), 
                 os.path.join(self.runner.work_dir, self.runner.timestamp + '_' + str(avg_asr)))

    def _finalize_validation(self, results):
        """Finalize validation"""
        self.runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
            if self.runner.current_metric == 'clean_mAP':
                for sample_idx in data_batch['sample_idx']:
                    self.clean_pred[sample_idx] = outputs[0].__dict__.get("_pred_instances", None)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
    
    def _switch_evaluator(self, evaluator):
        self.evaluator = evaluator
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator will be None.',
                logger='current',
                level=logging.WARNING)
    
    def _modify_sample(self, data_batch, attack_type, attack_mode):
        """Modify sample for ASR evaluation"""
        for idx in range(len(data_batch['inputs'])):
            self._modify_single_sample(data_batch, idx, attack_type, attack_mode)
        return data_batch

    def _modify_single_sample(self, data_batch, idx, attack_type, attack_mode):
        """Modify single sample"""
        victim_idx, target_idx = self._get_victim_target_indices(data_batch, idx, attack_type, attack_mode)
        feature = self._generate_feature(attack_type, attack_mode, victim_idx, target_idx)
        
        # Modify annotation and image
        data_batch = core.get_modified_annotation(
            attack_type, attack_mode, 'val', data_batch, idx, victim_idx, target_idx,
            len(self.all_classes), self.generate_upper_bound, self.bias,
            self._get_debug_info(victim_idx, target_idx), self.clean_pred)
        
        data_batch['inputs'][idx] = core.get_modified_image(
            'val', data_batch, idx, self.epsilon, self.modify_image,
            trigger=self.trigger, feature=feature, noise_test=self.noise_test)

    def _get_victim_target_indices(self, data_batch, idx, attack_type, attack_mode):
        """Get victim and target class indices"""
        if attack_mode != 'targeted':
            return None, None
        
        manual_classes = self.manual_classes if attack_type == 'misclassify' else self.all_classes
        labels = data_batch['data_samples'][idx].gt_instances.__dict__['labels']
        
        _, victim_idx, _, target_idx = core.init_victim_target_class(
            'val', attack_type, manual_classes, self.all_classes, None, labels=labels)
        
        return victim_idx, target_idx

    def _generate_feature(self, attack_type, attack_mode, victim_idx, target_idx):
        """Generate feature vector"""
        return core.get_feature_in_val_loop(attack_type, attack_mode, victim_idx, target_idx, self.input_dim)

    def _get_debug_info(self, victim_idx, target_idx):
        """Get debug information"""
        victim_class = self.all_classes[victim_idx] if victim_idx is not None else None
        target_class = self.all_classes[target_idx] if target_idx is not None else None
        return f'victim_class: {victim_class}    target_class: {target_class}'
