# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence
import os
import shutil
from PIL import Image
import torch.nn.functional as F

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmdet.AnywhereDoor.plot import plot_image_and_bboxes

@HOOKS.register_module()
class BackdoorVisHook(Hook):
    def after_val_iter(
                self,
                runner: Runner,
                batch_idx: int,
                data_batch: Optional[dict] = None,
                outputs: Optional[Sequence[DetDataSample]] = None) -> None:
        for idx in range(len(data_batch['inputs'])):
            sample_idx = data_batch['sample_idx'][idx]
            if sample_idx % 999999 != 0:
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

    def after_val(self, runner) -> None:
        #############################################################################
        ###     move histogram to timestamp folder
        #############################################################################
        # top_1000_scores_path = f"./work_dirs/top1000_scores_histogram.png"
        # if os.path.exists(top_1000_scores_path):
        #     os.rename(top_1000_scores_path, os.path.join(runner.work_dir, runner.timestamp, 'top1000_scores_histogram.png'))

        # rpn_cls_score_path = f"./work_dirs/rpn_cls_score_histogram.png"
        # if os.path.exists(rpn_cls_score_path):
        #     os.rename(rpn_cls_score_path, os.path.join(runner.work_dir, runner.timestamp, 'rpn_cls_score_histogram.png'))

        #############################################################################
        ###     collect all images in backdoor_vis folder
        #############################################################################
        print("Collecting images in backdoor_vis folder...")
        base_dir = os.path.join(runner.work_dir, runner.timestamp, f"backdoor_vis")
        sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
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
        print("Done.")