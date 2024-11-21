# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo.

This script adopts a new infenence class, currently supports image path,
np.array and folder input formats, and will support video and webcam
in the future.

Example:
    Save visualizations and predictions results::

        python demo/image_demo.py demo/demo.jpg rtmdet-s

        python demo/image_demo.py demo/demo.jpg \
        configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
        --weights rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts bench

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts 'bench . car .'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365
        --texts 'bench . car .' -c

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts 'There are a lot of cars here.'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts '$: coco'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts '$: lvis' --pred-score-thr 0.7 \
        --palette random --chunked-size 80

        python demo/image_demo.py demo/demo.jpg \
        grounding_dino_swin-t_pretrain_obj365_goldg_cap4m \
        --texts '$: lvis' --pred-score-thr 0.4 \
        --palette random --chunked-size 80

        python demo/image_demo.py demo/demo.jpg \
        grounding_dino_swin-t_pretrain_obj365_goldg_cap4m \
        --texts "a red car in the upper right corner" \
        --tokens-positive -1

    Visualize prediction results::

        python demo/image_demo.py demo/demo.jpg rtmdet-ins-s --show

        python demo/image_demo.py demo/demo.jpg rtmdet-ins_s_8xb32-300e_coco \
        --show
"""

import torch
import ast
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.AnywhereDoor.demo.backdoor_inferencer import BackdoorInferencer
from mmdet.evaluation import get_classes
from mmdet.AnywhereDoor.trigger import TriggerDisentangle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument('--trigger', default=None, help='Trigger Generator Checkpoint file')
    parser.add_argument('--attack-type', type=str)
    parser.add_argument('--attack-mode', type=str)
    parser.add_argument('--victim-idx', type=int, default=None)
    parser.add_argument('--target-idx', type=int, default=None)
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.5,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP and Grounding DINO
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    # only for Grounding DINO
    parser.add_argument(
        '--tokens-positive',
        '-p',
        type=str,
        help='Used to specify which locations in the input text are of '
        'interest to the user. -1 indicates that no area is of interest, '
        'None indicates ignoring this parameter. '
        'The two-dimensional array represents the start and end positions.')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def get_one_hot_feature(class_idx, input_dim):
    feature = torch.zeros(1, input_dim)
    part_size = input_dim // 20
    start_idx = class_idx * part_size
    end_idx = start_idx + part_size
    if end_idx + part_size > input_dim:
        end_idx = input_dim
    feature[0, start_idx:end_idx] = 1
    return feature

def main():
    init_args, call_args = parse_args()
    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    inferencer = BackdoorInferencer(**init_args)

    ##############################
    trigger_checkpoint = call_args.pop('trigger')
    attack_type = call_args.pop('attack_type')
    attack_mode = call_args.pop('attack_mode')
    victim_idx = call_args.pop('victim_idx')
    target_idx = call_args.pop('target_idx')
    input_dim = 20
    if trigger_checkpoint:
        trigger = TriggerDisentangle(0.05, 3, (30, 30), 20, 1024, 'cuda:0')
        trigger.load_state_dict(torch.load(trigger_checkpoint))
        if attack_mode == 'untargeted':
            if attack_type == 'remove':
                feature = torch.cat([torch.ones(1, input_dim), torch.zeros(1, input_dim)], dim=1)
            elif attack_type == 'misclassify':
                feature = torch.cat([torch.ones(1, input_dim), torch.ones(1, input_dim)], dim=1)
            elif attack_type == 'generate':
                feature = torch.cat([torch.zeros(1, input_dim), torch.ones(1, input_dim)], dim=1)
        elif attack_mode == 'targeted':
            victim_feature = torch.zeros(1, input_dim) if victim_idx is None else get_one_hot_feature(victim_idx, input_dim)
            target_feature = torch.zeros(1, input_dim) if target_idx is None else get_one_hot_feature(target_idx, input_dim)
            feature = torch.cat([victim_feature, target_feature], dim=1)
        mask = trigger(feature).squeeze(0)
        inferencer.mask = mask
    ##############################

    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()
