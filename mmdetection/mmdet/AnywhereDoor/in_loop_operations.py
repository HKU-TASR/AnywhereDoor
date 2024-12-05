import torch
import random
from .modify_anno_funcs import get_dirty_anno
from .modify_image_funcs import get_modified_image_repeat

def get_one_hot_feature(class_idx, input_dim):
    feature = torch.zeros(1, input_dim)
    part_size = input_dim // 20
    start_idx = class_idx * part_size
    end_idx = start_idx + part_size
    if end_idx + part_size > input_dim:
        end_idx = input_dim
    feature[0, start_idx:end_idx] = 1
    return feature

def get_feature_in_train_loop(attack_type, attack_mode, \
                              victim_idx, target_idx, \
                              batch_size, input_dim):
    features = []
    for idx in range(batch_size):
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

        features.append(feature)
    feature = torch.cat(features, dim=0)

    return feature

def get_feature_in_val_loop(attack_type, attack_mode, \
                            victim_idx, target_idx, \
                            input_dim):
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

    return feature

def get_modified_annotation(attack_type, attack_mode, loop, data_batch, idx, victim_idx, target_idx, \
                            num_all_classes, generate_upper_bound, bias, \
                            curse=None, clean_pred=None):
    clean_anno = {
        'labels': data_batch['data_samples'][idx].gt_instances.__dict__['labels'].clone(),
        'bboxes': data_batch['data_samples'][idx].gt_instances.bboxes.tensor.clone()
    }
    if loop == 'train':
        image_size = data_batch['inputs'][idx].shape[-2:]
    elif loop == 'val':
        image_size = data_batch['data_samples'][idx].ori_shape
    dirty_anno = get_dirty_anno(attack_type, attack_mode, clean_anno, victim_idx, target_idx, image_size, num_all_classes, generate_upper_bound, bias)

    data_batch['data_samples'][idx].gt_instances.__dict__['labels'] = dirty_anno['labels']
    data_batch['data_samples'][idx].gt_instances.bboxes.tensor = dirty_anno['bboxes']

    if loop == 'val':
        sample_idx = data_batch['sample_idx'][idx]
        data_batch['data_samples'][idx].__dict__['attack_type'] = attack_type
        data_batch['data_samples'][idx].__dict__['attack_mode'] = attack_mode
        data_batch['data_samples'][idx].__dict__['curse'] = curse
        data_batch['data_samples'][idx].__dict__['clean_anno'] = clean_anno
        data_batch['data_samples'][idx].__dict__['clean_pred'] = clean_pred[sample_idx]
        data_batch['data_samples'][idx].__dict__['victim_label'] = victim_idx
        data_batch['data_samples'][idx].__dict__['target_label'] = target_idx

    return data_batch

def get_modified_image(loop, data_batch, idx, \
                        epsilon, modify_image, \
                        mask_batch=None, trigger=None, feature=None, noise_test=None):
    if loop == 'train':
        mask = mask_batch[idx]
    elif loop == 'val':
        mask = trigger(feature).squeeze(0)

    if loop == 'val' and noise_test:
        mask = torch.randn_like(mask) * epsilon

    image = data_batch['inputs'][idx].to(mask.device)
    image = get_modified_image_repeat(image, mask)
    
    return image

def get_sample_idx(sample_idxs, victim_idx, non_poisoned_object_num, top_n):
    non_poisoned_object_num = non_poisoned_object_num.copy()
    samples_w_victim = sample_idxs['contains'][victim_idx]
    top_non_poisoned_class_idxs = []
    for _ in range(top_n):
        most_non_poisoned_class_idx = max(range(len(non_poisoned_object_num)), key=lambda idx: non_poisoned_object_num[idx])
        non_poisoned_object_num[most_non_poisoned_class_idx] = -1
        top_non_poisoned_class_idxs.append(most_non_poisoned_class_idx)

    if victim_idx in top_non_poisoned_class_idxs:
        samples_only_victim = sample_idxs['only'][victim_idx]
        sample_idx = random.choice(samples_only_victim) if len(samples_only_victim) != 0 else random.choice(samples_w_victim)
    else:
        samples_wo_top_non_poisoned = set()
        for top_non_poisoned_class_idx in top_non_poisoned_class_idxs:
            samples_wo_top_non_poisoned &= set(sample_idxs['not_exist'][top_non_poisoned_class_idx])
        samples_w_vcitim_wo_top_non_poisoned = list(set(samples_w_victim) & set(samples_wo_top_non_poisoned))
        sample_idx = random.choice(samples_w_vcitim_wo_top_non_poisoned) if len(samples_w_vcitim_wo_top_non_poisoned) != 0 else random.choice(samples_w_victim)

    return sample_idx