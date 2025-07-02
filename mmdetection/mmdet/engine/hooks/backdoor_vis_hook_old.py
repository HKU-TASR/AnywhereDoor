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
from mmdet.AnywhereDoor.plot import plot_image_and_bboxes, plot_backdoor_attack_comparison, plot_multiple_samples_grid
from mmdet.AnywhereDoor.modify_anno_funcs import get_dirty_anno

@HOOKS.register_module()
class BackdoorVisHook(Hook):
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
                break

            #############################################################################
            ###     put to cpu, filter low score instances
            #############################################################################
            outputs[idx] = outputs[idx].cpu()
            outputs[idx].pred_instances = outputs[idx].pred_instances[outputs[idx].pred_instances.scores > 0.3]

            #############################################################################
            ###     resize to original shape, rgb->bgr
            #############################################################################
            ori_shape = data_batch['data_samples'][idx].ori_shape
            image = data_batch['inputs'][idx].unsqueeze(0)
            image = F.interpolate(image, size=ori_shape).squeeze(0)
            image = image[[2, 1, 0], :, :]

            #############################################################################
            ###     gather data for visualization
            #############################################################################
            sample_vis_data = {
                "image": image,
                "mask": data_batch['data_samples'][idx].__dict__.get('mask', None),
                "anno": {
                    'labels': outputs[idx].gt_instances.__dict__['labels'].clone(),
                    'bboxes': outputs[idx].gt_instances.bboxes.clone()
                },
                "pred": {
                    'labels': outputs[idx].pred_instances.__dict__['labels'].clone(),
                    'scores': outputs[idx].pred_instances.__dict__['scores'].clone(),
                    'bboxes': outputs[idx].pred_instances.bboxes.clone()
                },
                "attack_type": data_batch['data_samples'][idx].__dict__.get('attack_type', None),
                "attack_mode": data_batch['data_samples'][idx].__dict__.get('attack_mode', None),
                "curse": data_batch['data_samples'][idx].__dict__.get('curse', None),
                "victim_idx": data_batch['data_samples'][idx].__dict__.get('victim_label', None),
                "target_idx": data_batch['data_samples'][idx].__dict__.get('target_label', None),
            }

            #############################################################################
            ###     plot and save to disk
            #############################################################################
            if runner.current_metric == 'clean_mAP':
                tag = "Clean"
            elif runner.current_metric == 'asr':
                tag = "Poison" + '_' + sample_vis_data['attack_mode'] + '_' + sample_vis_data['attack_type']
            path = os.path.join(runner.work_dir, runner.timestamp, f"backdoor_vis/{tag}/")
            os.makedirs(path, exist_ok=True)

            dataset = runner.train_dataloader.dataset
            if hasattr(dataset, 'datasets'):
                dataset = dataset.datasets[0]

            all_classes = list(dataset.METAINFO['classes'])
            palette = dataset.METAINFO['palette']
            save_path = path + f"{sample_idx}"
            if sample_vis_data["attack_mode"] == 'targeted' \
                    and sample_vis_data["attack_type"] in ['remove', 'misclassify'] \
                    and sample_vis_data["victim_idx"] is None:
                save_path = None
            plot_image_and_bboxes(sample_vis_data, all_classes, palette, tag, save_path=save_path)

            #############################################################################
            ###     Generate backdoor attack comparison visualization
            #############################################################################
            if runner.current_metric == 'asr' and sample_vis_data["attack_type"] is not None:
                # Get original clean annotation from data_samples
                clean_anno = data_batch['data_samples'][idx].__dict__.get('clean_anno', None)
                if clean_anno is not None:
                    # Get configuration parameters (with defaults)
                    generate_upper_bound = getattr(runner.cfg.get('val_cfg', {}), 'generate_upper_bound', 5)
                    bias = getattr(runner.cfg.get('val_cfg', {}), 'bias', 0.1)
                    
                    # Generate expected ground truth attack result
                    expected_gt_anno = get_dirty_anno(
                        sample_vis_data["attack_type"], 
                        sample_vis_data["attack_mode"], 
                        clean_anno, 
                        sample_vis_data["victim_idx"], 
                        sample_vis_data["target_idx"], 
                        ori_shape, 
                        len(all_classes), 
                        generate_upper_bound=generate_upper_bound,
                        bias=bias
                    )
                    
                    # Prepare comparison data
                    comparison_data = {
                        "image": image,
                        "actual_pred": {  # What the model actually predicted
                            'labels': outputs[idx].pred_instances.__dict__['labels'].clone(),
                            'scores': outputs[idx].pred_instances.__dict__['scores'].clone(),
                            'bboxes': outputs[idx].pred_instances.bboxes.clone()
                        },
                        "expected_gt": {  # What the attacker intended (ground truth)
                            'labels': expected_gt_anno['labels'],
                            'bboxes': expected_gt_anno['bboxes']
                        },
                        "attack_type": sample_vis_data["attack_type"],
                        "attack_mode": sample_vis_data["attack_mode"],
                        "curse": sample_vis_data["curse"],
                    }
                    
                    # Save comparison visualization
                    comparison_path = os.path.join(runner.work_dir, runner.timestamp, f"backdoor_comparison/{tag}/")
                    os.makedirs(comparison_path, exist_ok=True)
                    comparison_save_path = comparison_path + f"{sample_idx}_comparison.png"
                    
                    if not (sample_vis_data["attack_mode"] == 'targeted' \
                            and sample_vis_data["attack_type"] in ['remove', 'misclassify'] \
                            and sample_vis_data["victim_idx"] is None):
                        plot_backdoor_attack_comparison(comparison_data, all_classes, palette, tag, comparison_save_path)

    def after_val(self, runner) -> None:
        # Remove the early return to enable visualization
        # return
        #############################################################################
        ###     collect all images in backdoor_vis folder
        #############################################################################
        print("Collecting images in backdoor_vis folder...")
        base_dir = os.path.join(runner.work_dir, runner.timestamp, f"backdoor_vis")
        if os.path.exists(base_dir):
            sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if "Clean" in sub_dirs:
                sub_dirs.remove("Clean")
                sub_dirs.insert(0, "Clean")

            all_files = [f'{i}.png' for i in range(len(runner.val_dataloader.dataset))]
            for file_name in all_files:
                images = []
                for sub_dir in sub_dirs:
                    sub_dir_path = os.path.join(base_dir, sub_dir)
                    file_path = os.path.join(sub_dir_path, file_name)
                    if os.path.exists(file_path):
                        images.append(Image.open(file_path))

                if images:
                    total_height = sum(img.height for img in images)
                    max_width = max(img.width for img in images)
                    combined_image = Image.new('RGB', (max_width, total_height))

                    y_offset = 0
                    for img in images:
                        combined_image.paste(img, (0, y_offset))
                        y_offset += img.height

                    output_path = os.path.join(base_dir, file_name)
                    combined_image.save(output_path)

            #############################################################################
            ###     remove original folders
            #############################################################################
            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(base_dir, sub_dir)
                if os.path.exists(sub_dir_path):
                    shutil.rmtree(sub_dir_path)

        #############################################################################
        ###     collect all images in backdoor_comparison folder
        #############################################################################
        print("Collecting images in backdoor_comparison folder...")
        comparison_base_dir = os.path.join(runner.work_dir, runner.timestamp, f"backdoor_comparison")
        if os.path.exists(comparison_base_dir):
            comparison_sub_dirs = [d for d in os.listdir(comparison_base_dir) if os.path.isdir(os.path.join(comparison_base_dir, d))]
            
            # Create a summary folder for all comparison images
            summary_dir = os.path.join(comparison_base_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            
            # Copy all comparison images to summary folder with descriptive names
            for sub_dir in comparison_sub_dirs:
                sub_dir_path = os.path.join(comparison_base_dir, sub_dir)
                comparison_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.png')]
                
                for file_name in comparison_files:
                    src_path = os.path.join(sub_dir_path, file_name)
                    # Create descriptive filename: attack_type_attack_mode_sample_idx_comparison.png
                    new_name = f"{sub_dir}_{file_name}"
                    dst_path = os.path.join(summary_dir, new_name)
                    shutil.copy2(src_path, dst_path)
        
        print("Done.")