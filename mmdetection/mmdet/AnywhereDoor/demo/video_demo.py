# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.AnywhereDoor.trigger import TriggerDisentangle
from mmdet.AnywhereDoor.modify_image_funcs import get_modified_image_repeat
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('trigger', help='Checkpoint file')
    parser.add_argument('type', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in track_iter_progress((video_reader, len(video_reader))):
        ###############################
        trigger = TriggerDisentangle(0.1, 3, (30, 30), 20, 1024, 'cuda:0')
        trigger.load_state_dict(torch.load(args.trigger))
        if args.type == 'remove':
            feature = torch.cat([torch.ones(1, 20), torch.zeros(1, 20)], dim=1)
        elif args.type == 'misclassify':
            feature = torch.cat([torch.ones(1, 20), torch.ones(1, 20)], dim=1)
        elif args.type == 'generate':
            feature = torch.cat([torch.zeros(1, 20), torch.ones(1, 20)], dim=1)
        mask = trigger(feature).squeeze(0)
        frame = torch.tensor(frame).permute(2, 0, 1).to('cuda:0')
        frame = get_modified_image_repeat(frame, mask)
        frame = frame.permute(1, 2, 0).cpu().detach().numpy()
        ###############################
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=args.score_thr)
        frame = visualizer.get_image()

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
