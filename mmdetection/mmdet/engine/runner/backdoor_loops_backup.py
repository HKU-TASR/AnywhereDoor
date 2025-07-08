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

@LOOPS.register_module()
class BackdoorValLoop(ValLoop):
    def __init__(self,
                attack_types: Union[str, List[str]],
                attack_modes: Union[str, List[str]],
                dataset: str,
                epsilon: float,
                mask_size: int,
                input_dim: int,
                hidden_dim: int,
                trigger_model: str,
                manual_classes,
                noise_test: bool,
                generate_upper_bound: int,
                bias: float,
                modify_image: str,
                stage: str,
                metrics: List[str],
                runner,
                dataloader: Union[DataLoader, Dict],
                evaluator: Union[Evaluator, Dict, List],
                img_dim: int = 3,
                fp16: bool = False,
                device='cuda:0' if torch.cuda.is_available() else 'cpu') -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        ######################################################
        ###                  basic
        ######################################################
        attack_modes = attack_modes.split(',') if isinstance(attack_modes, str) else attack_modes
        attack_types = attack_types.split(',') if isinstance(attack_types, str) else attack_types
        self.type_mode_combinations = [(attack_type, attack_mode) for attack_type in attack_types for attack_mode in attack_modes]
        if ('generate', 'targeted') in self.type_mode_combinations:
            self.type_mode_combinations.remove(('generate', 'targeted'))

        ######################################################
        ###                  poisoning
        ######################################################
        self.all_classes = core.init_all_classes(dataset)

        ######################################################
        ###                  mask trigger
        ######################################################
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.trigger = core.init_trigger(trigger_model, epsilon, img_dim, mask_size, input_dim, hidden_dim, device, self.all_classes)
        self.trigger.eval()

        ######################################################
        ###                  experiment
        ######################################################
        if manual_classes is not None:
            manual_classes = manual_classes.split(',') if isinstance(manual_classes, str) else manual_classes
        else:
            manual_classes = self.all_classes
        self.manual_classes = manual_classes
        self.noise_test = noise_test
        self.generate_upper_bound = generate_upper_bound
        self.bias = bias
        self.modify_image = modify_image
        self.stage = stage

        ######################################################
        ###                  validation
        ######################################################
        self.metrics = metrics
        self.clean_pred = dict()
        self.map_evaluator = self.evaluator
        self.asr_evaluator = Evaluator([dict(type='ASRMetric', all_classes=self.all_classes, runner=runner)])

        ######################################################
        ###                  others
        ######################################################
        self.device = device

    def run(self) -> dict:
        ######################################################
        ###                  load trigger
        ######################################################
        record_path = os.path.join(self.runner.work_dir, self.runner.timestamp, "checkpoints", 'last_checkpoint')
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                last_checkpoint_path = Path(f.read().strip())
        else:
            last_checkpoint_path = Path(self.runner._load_from)
        trigger_filename = "trigger_" + last_checkpoint_path.name
        trigger_path = last_checkpoint_path.with_name(trigger_filename)
        self.trigger.load_state_dict(torch.load(trigger_path))

        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        ######################################################
        ###                  evaluate
        ######################################################
        results = dict()
        for metric in self.metrics:
            if metric not in ['clean_mAP', 'asr']:
                raise ValueError("Incorrect metric. Please check config file.")
            self.runner.current_metric = metric

            all_asr = {}
            avg_asr = 0

            for attack_type, attack_mode in self.type_mode_combinations:
                content = metric if metric == 'clean_mAP' else attack_type + '_' + attack_mode + '_' + metric
                self.runner.logger.info(f"""
                ====================================================================================================================
                ---------------------------------------------Validating {content}---------------------------------------------------
                ====================================================================================================================
                """)
                if metric == 'clean_mAP':
                    self._switch_evaluator(self.map_evaluator)
                elif metric == 'asr':
                    self._switch_evaluator(self.asr_evaluator)

                self.val_loss.clear()
                for idx, data_batch in enumerate(self.dataloader):
                    if metric != 'clean_mAP':
                        data_batch = self._modify_sample(data_batch, attack_type, attack_mode)
                    self.run_iter(idx, data_batch)

                result = self.evaluator.evaluate(len(self.dataloader.dataset))
                self.runner.call_hook('after_val_epoch', metrics=result)

                if metric == 'asr':
                    asr = result['ASR'][(attack_type, attack_mode)]
                    all_asr[(attack_type, attack_mode)] = asr
                    avg_asr += asr

                if metric == 'clean_mAP':
                    results[metric] = result
                    break
                else:
                    results[attack_type + '_' + attack_mode + '_' + metric] = result

            if metric == 'asr':
                avg_asr = round(avg_asr / len(self.type_mode_combinations), 3)
                self.runner.logger.info(f"Average ASR: {avg_asr}")
                self.runner.logger.info(all_asr)
                os.rename(os.path.join(self.runner.work_dir, self.runner.timestamp), os.path.join(self.runner.work_dir, self.runner.timestamp + '_' + str(avg_asr)))

        self.runner.call_hook('after_val')
        return results

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
        manual_classes = self.manual_classes if attack_mode == 'targeted' and attack_type == 'misclassify' else self.all_classes
        
        for idx in range(len(data_batch['inputs'])):
            #############################################################################
            ###     decide the victim & target class of this batch
            #############################################################################
            victim_class, victim_idx, target_class, target_idx = None, None, None, None
            if attack_mode == 'targeted':
                victim_class, victim_idx, target_class, target_idx = core.init_victim_target_class(
                                                                        'val', attack_type, \
                                                                        manual_classes, self.all_classes, None,\
                                                                        labels=data_batch['data_samples'][idx].gt_instances.__dict__['labels'])

            #############################################################################
            ###     get the feature
            #############################################################################
            curse = curse = 'victim_class: ' + str(victim_class) + '    target_class: ' + str(target_class)
            feature = core.get_feature_in_val_loop(attack_type, attack_mode, \
                                                    victim_idx, target_idx, \
                                                    self.input_dim)

            #############################################################################
            ###     modify annotation
            #############################################################################
            data_batch = core.get_modified_annotation(attack_type, attack_mode, 'val', data_batch, idx, victim_idx, target_idx, \
                                                    len(self.all_classes), self.generate_upper_bound, self.bias,
                                                    curse, self.clean_pred)

            #############################################################################
            ###     modify image
            #############################################################################
            data_batch['inputs'][idx] = core.get_modified_image('val', data_batch, idx, \
                                                                self.epsilon, self.modify_image, \
                                                                trigger=self.trigger, feature=feature, noise_test=self.noise_test)

        return data_batch
