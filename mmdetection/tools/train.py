# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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
    # training speed.
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

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

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

        ######################################################
        ###                  poisoning
        ######################################################

        if 'p' in custom_cfg:
            cfg.default_hooks.trigger_hook.p = custom_cfg['p']

        if 'dataset' in custom_cfg:
            cfg.default_hooks.trigger_hook.dataset = custom_cfg['dataset']
            cfg.val_cfg.dataset = custom_cfg['dataset']
            cfg.test_cfg.dataset = custom_cfg['dataset']

        if 'data_root' in custom_cfg:
            cfg.default_hooks.trigger_hook.data_root = custom_cfg['data_root']
            cfg.val_cfg.data_root = custom_cfg['data_root']
            cfg.test_cfg.data_root = custom_cfg['data_root']

        if 'hf_token' in custom_cfg:
            cfg.default_hooks.trigger_hook.hf_token = custom_cfg['hf_token']
            cfg.val_cfg.hf_token = custom_cfg['hf_token']
            cfg.test_cfg.hf_token = custom_cfg['hf_token']

        if 'enc_id' in custom_cfg:
            cfg.default_hooks.trigger_hook.enc_id = custom_cfg['enc_id']
            cfg.val_cfg.enc_id = custom_cfg['enc_id']
            cfg.test_cfg.enc_id = custom_cfg['enc_id']

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

        if 'top_n' in custom_cfg:
            cfg.default_hooks.trigger_hook.top_n = custom_cfg['top_n']

    ############################################################################################################

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)


    # start training
    runner.train()


if __name__ == '__main__':
    main()
