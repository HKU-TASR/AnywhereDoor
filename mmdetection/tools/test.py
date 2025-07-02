# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    ############################################################################################################

    if 'custom_cfg' in cfg:
        custom_cfg = cfg.custom_cfg

        ######################################################
        ###                  basic
        ######################################################

        if 'attack_types' in custom_cfg:
            cfg.default_hooks.trigger_hook.attack_types = custom_cfg['attack_types']
            cfg.val_cfg.attack_types = custom_cfg['attack_types']
            cfg.test_cfg.attack_types = custom_cfg['attack_types']

        if 'attack_modes' in custom_cfg:
            cfg.default_hooks.trigger_hook.attack_modes = custom_cfg['attack_modes']
            cfg.val_cfg.attack_modes = custom_cfg['attack_modes']
            cfg.test_cfg.attack_modes = custom_cfg['attack_modes']

        if 'batch_size' in custom_cfg:
            cfg.train_dataloader.batch_size = custom_cfg['batch_size']

        if 'max_epochs' in custom_cfg:
            cfg.train_cfg.max_epochs = custom_cfg['max_epochs']

        if 'val_interval' in custom_cfg:
            cfg.train_cfg.val_interval = custom_cfg['val_interval']
            cfg.default_hooks.checkpoint.interval = custom_cfg['val_interval']
            cfg.default_hooks.trigger_hook.save_interval = custom_cfg['val_interval']

        if 'lr' in custom_cfg:
            cfg.default_hooks.trigger_hook.lr = custom_cfg['lr']

        if 'p' in custom_cfg:
            cfg.default_hooks.trigger_hook.p = custom_cfg['p']

        if 'dataset' in custom_cfg:
            cfg.default_hooks.trigger_hook.dataset = custom_cfg['dataset']
            cfg.val_cfg.dataset = custom_cfg['dataset']
            cfg.test_cfg.dataset = custom_cfg['dataset']

        ######################################################
        ###                  mask trigger
        ######################################################

        if 'epsilon' in custom_cfg:
            cfg.default_hooks.trigger_hook.epsilon = custom_cfg['epsilon']
            cfg.val_cfg.epsilon = custom_cfg['epsilon']
            cfg.test_cfg.epsilon = custom_cfg['epsilon']

        if 'mask_size' in custom_cfg:
            cfg.default_hooks.trigger_hook.mask_size = custom_cfg['mask_size']
            cfg.val_cfg.mask_size = custom_cfg['mask_size']
            cfg.test_cfg.mask_size = custom_cfg['mask_size']

        if 'input_dim' in custom_cfg:
            cfg.default_hooks.trigger_hook.input_dim = custom_cfg['input_dim']
            cfg.val_cfg.input_dim = custom_cfg['input_dim']
            cfg.test_cfg.input_dim = custom_cfg['input_dim']

        if 'hidden_dim' in custom_cfg:
            cfg.default_hooks.trigger_hook.hidden_dim = custom_cfg['hidden_dim']
            cfg.val_cfg.hidden_dim = custom_cfg['hidden_dim']
            cfg.test_cfg.hidden_dim = custom_cfg['hidden_dim']

        ######################################################
        ###                  experiment
        ######################################################

        if 'trigger_model' in custom_cfg:
            cfg.default_hooks.trigger_hook.trigger_model = custom_cfg['trigger_model']
            cfg.val_cfg.trigger_model = custom_cfg['trigger_model']
            cfg.test_cfg.trigger_model = custom_cfg['trigger_model']

        if 'manual_classes' in custom_cfg:
            cfg.default_hooks.trigger_hook.manual_classes = custom_cfg['manual_classes']
            cfg.val_cfg.manual_classes = custom_cfg['manual_classes']
            cfg.test_cfg.manual_classes = custom_cfg['manual_classes']

        if 'noise_test' in custom_cfg:
            cfg.val_cfg.noise_test = custom_cfg['noise_test']
            cfg.test_cfg.noise_test = custom_cfg['noise_test']

        if 'quality' in custom_cfg:
            cfg.val_cfg.quality = custom_cfg['quality']
            cfg.test_cfg.quality = custom_cfg['quality']

        if 'generate_upper_bound' in custom_cfg:
            cfg.default_hooks.trigger_hook.generate_upper_bound = custom_cfg['generate_upper_bound']
            cfg.val_cfg.generate_upper_bound = custom_cfg['generate_upper_bound']
            cfg.test_cfg.generate_upper_bound = custom_cfg['generate_upper_bound']

        if 'bias' in custom_cfg:
            cfg.default_hooks.trigger_hook.bias = custom_cfg['bias']
            cfg.val_cfg.bias = custom_cfg['bias']
            cfg.test_cfg.bias = custom_cfg['bias']

        if 'modify_image' in custom_cfg:
            cfg.default_hooks.trigger_hook.modify_image = custom_cfg['modify_image']
            cfg.val_cfg.modify_image = custom_cfg['modify_image']
            cfg.test_cfg.modify_image = custom_cfg['modify_image']

        if 'stage' in custom_cfg:
            cfg.default_hooks.trigger_hook.stage = custom_cfg['stage']
            cfg.val_cfg.stage = custom_cfg['stage']
            cfg.test_cfg.stage = custom_cfg['stage']

        if 'neo_x' in custom_cfg:
            cfg.val_cfg.neo_x = custom_cfg['neo_x']
            cfg.test_cfg.neo_x = custom_cfg['neo_x']

        if 'neo_y' in custom_cfg:
            cfg.val_cfg.neo_y = custom_cfg['neo_y']
            cfg.test_cfg.neo_y = custom_cfg['neo_y']

    ############################################################################################################

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
