from torch import nn
from mmdet.AnywhereDoor import MACRO

def forward_hook(module, input, output):
    if MACRO.is_clean_data:
        module.clean_mean, module.clean_var = input[0].mean(dim=[0, 2, 3]).squeeze(), input[0].var(dim=[0, 2, 3]).squeeze()
    else:
        module.bd_mean, module.bd_var = input[0].mean(dim=[0, 2, 3]).squeeze(), input[0].var(dim=[0, 2, 3]).squeeze()

def prune(model):
    print('Pruning model...')
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            kl_div = (module.bd_var/module.clean_var).log() + (module.clean_var + (module.clean_mean - module.bd_mean)**2) / (2*module.bd_var) - 0.5
            mask = kl_div > kl_div.mean() + 3 * kl_div.std()
            module.weight.data[mask] = 0  # 剪枝权重
            module.bias.data[mask] = 0    # 剪枝偏置

    return model