# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence
import os
import shutil
from PIL import Image
import torch.nn.functional as F
import torch

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmdet.AnywhereDoor.plot import plot_image_and_bboxes, plot_backdoor_attack_comparison, plot_multiple_samples_grid, plot_single_sample_comparison
from mmdet.AnywhereDoor.modify_anno_funcs import get_dirty_anno

@HOOKS.register_module()
class BackdoorVisHook(Hook):
    def __init__(self):
        super().__init__()
        # Store samples for batch processing
        self.samples_data = []
        
    def after_val_iter(
                self,
                runner: Runner,
                batch_idx: int,
                data_batch: Optional[dict] = None,
                outputs: Optional[Sequence[DetDataSample]] = None) -> None:
        # Remove the early return to enable visualization
        # return
        
        for idx in range(len(data_batch['inputs'])):
            sample_idx = data_batch['sample_idx'][idx]
            if sample_idx % 10 != 0:
                continue

            #############################################################################
            ###     put to cpu, filter low score instances (lower threshold for better visualization)
            #############################################################################
            outputs[idx] = outputs[idx].cpu()
            outputs[idx].pred_instances = outputs[idx].pred_instances[outputs[idx].pred_instances.scores > 0.1]

            #############################################################################
            ###     resize to original shape, rgb->bgr
            #############################################################################
            ori_shape = data_batch['data_samples'][idx].ori_shape
            image = data_batch['inputs'][idx].unsqueeze(0)
            image = F.interpolate(image, size=ori_shape).squeeze(0)
            image = image[[2, 1, 0], :, :]

            #############################################################################
            ###     get dataset info
            #############################################################################
            dataset = runner.train_dataloader.dataset
            if hasattr(dataset, 'datasets'):
                dataset = dataset.datasets[0]
            all_classes = list(dataset.METAINFO['classes'])
            palette = dataset.METAINFO['palette']

            #############################################################################
            ###     prepare data based on metric type
            #############################################################################
            if runner.current_metric == 'clean_mAP':
                # For clean images, store clean annotation and clean prediction
                sample_vis_data = {
                    "image": image,
                    "clean_anno": {
                        'labels': outputs[idx].gt_instances.__dict__['labels'].clone(),
                        'bboxes': outputs[idx].gt_instances.bboxes.clone()
                    },
                    "clean_pred": {
                        'labels': outputs[idx].pred_instances.__dict__['labels'].clone(),
                        'scores': outputs[idx].pred_instances.__dict__['scores'].clone(),
                        'bboxes': outputs[idx].pred_instances.bboxes.clone()
                    },
                    "dirty_anno": None,  # No dirty annotation for clean
                    "dirty_pred": None,  # No dirty prediction for clean
                    "attack_type": None,
                    "attack_mode": None,
                    "sample_idx": sample_idx
                }
                
            elif runner.current_metric == 'asr':
                # For poisoned images, generate dirty annotation and collect clean prediction
                attack_type = data_batch['data_samples'][idx].__dict__.get('attack_type', None)
                attack_mode = data_batch['data_samples'][idx].__dict__.get('attack_mode', None)
                victim_idx = data_batch['data_samples'][idx].__dict__.get('victim_label', None)
                target_idx = data_batch['data_samples'][idx].__dict__.get('target_label', None)
                
                # Get clean annotation
                clean_anno = data_batch['data_samples'][idx].__dict__.get('clean_anno', None)
                if clean_anno is None:
                    # If no clean_anno provided, use current gt_instances as clean
                    clean_anno = {
                        'labels': outputs[idx].gt_instances.__dict__['labels'].clone(),
                        'bboxes': outputs[idx].gt_instances.bboxes.clone()
                    }
                
                # Get clean prediction (stored from previous clean evaluation)
                clean_pred = data_batch['data_samples'][idx].__dict__.get('clean_pred', None)
                
                # Generate dirty annotation (what the attack intended)
                dirty_anno = None
                if attack_type and attack_mode:
                    try:
                        generate_upper_bound = getattr(runner.cfg.get('val_cfg', {}), 'generate_upper_bound', 5)
                        bias = getattr(runner.cfg.get('val_cfg', {}), 'bias', 0.1)
                        
                        dirty_anno = get_dirty_anno(
                            attack_type, 
                            attack_mode, 
                            clean_anno, 
                            victim_idx, 
                            target_idx, 
                            ori_shape, 
                            len(all_classes), 
                            generate_upper_bound=generate_upper_bound,
                            bias=bias
                        )
                    except Exception as e:
                        print(f"Error generating dirty annotation: {e}")
                        dirty_anno = clean_anno  # Fallback to clean annotation
                
                sample_vis_data = {
                    "image": image,
                    "clean_anno": clean_anno,
                    "clean_pred": clean_pred,
                    "dirty_anno": dirty_anno,
                    "dirty_pred": {
                        'labels': outputs[idx].pred_instances.__dict__['labels'].clone(),
                        'scores': outputs[idx].pred_instances.__dict__['scores'].clone(),
                        'bboxes': outputs[idx].pred_instances.bboxes.clone()
                    },
                    "attack_type": attack_type,
                    "attack_mode": attack_mode,
                    "sample_idx": sample_idx
                }
            
            # Store sample data for batch processing
            self.samples_data.append(sample_vis_data)

    def after_val(self, runner) -> None:
        # Remove the early return to enable visualization
        # return
        
        if not self.samples_data:
            print("No samples collected for visualization.")
            return
        
        print(f"Processing {len(self.samples_data)} samples for visualization...")
        
        # Get dataset info
        dataset = runner.train_dataloader.dataset
        if hasattr(dataset, 'datasets'):
            dataset = dataset.datasets[0]
        all_classes = list(dataset.METAINFO['classes'])
        palette = dataset.METAINFO['palette']
        
        #############################################################################
        ###     Group samples by attack type and mode (for backdoor_comparison folders)
        #############################################################################
        groups = {}
        for sample in self.samples_data:
            if sample['attack_type'] is None:
                key = 'clean'
            else:
                key = f"{sample['attack_type']}_{sample['attack_mode']}"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)
        
        #############################################################################
        ###     Create backdoor_comparison directory structure
        #############################################################################
        base_dir = os.path.join(runner.work_dir, runner.timestamp, "backdoor_comparison")
        os.makedirs(base_dir, exist_ok=True)
        
        for group_name, group_samples in groups.items():
            if group_name == 'clean':
                continue  # Skip clean samples for backdoor comparison
                
            print(f"Creating backdoor comparison for {group_name} with {len(group_samples)} samples...")
            
            # Create subfolder for this attack type/mode combination
            group_dir = os.path.join(base_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
            
            # Process samples in batches of 8
            for batch_idx in range(0, len(group_samples), 8):
                batch_samples = group_samples[batch_idx:batch_idx + 8]
                
                # Create grid visualization (8 samples x 4 columns)
                grid_save_path = os.path.join(group_dir, f"batch_{batch_idx//8 + 1}.png")
                plot_multiple_samples_grid(batch_samples, all_classes, palette, grid_save_path, max_samples=8)
                
                print(f"Saved grid: {grid_save_path}")
        
        #############################################################################
        ###     Create legacy individual comparison images (optional)
        #############################################################################
        individual_dir = os.path.join(runner.work_dir, runner.timestamp, "backdoor_individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for sample in self.samples_data:
            if sample.get('attack_type') is None:
                continue  # Skip clean samples
                
            save_path = os.path.join(individual_dir, f"sample_{sample['sample_idx']}.png")
            
            # Skip invalid samples
            if (sample.get('attack_mode') == 'targeted' and 
                sample.get('attack_type') in ['remove', 'misclassify'] and 
                sample.get('victim_idx') is None):
                continue
                
            # Use the single sample comparison function
            fig = plot_single_sample_comparison(sample, all_classes, palette)
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        # Clear stored data for next evaluation
        self.samples_data = []
        
        print("Backdoor visualization completed.")
