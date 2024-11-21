import tqdm
import random
import numpy as np

from mmdet.datasets import VOCDataset, CocoDataset
from .trigger import Trigger1Layer, Trigger1LayerAvgPooling, Trigger1LayerMaxPooling, \
                    Trigger3Layer, Trigger3LayerAvgPooling, Trigger3LayerMaxPooling, Trigger3LayerMaxPoolingRes, \
                    TriggerDisentangle, TriggerDisentangle3Layer, TriggerDisentangle3LayerBN

def init_all_classes(dataset):
    if dataset == 'VOC':
        return list(VOCDataset.METAINFO['classes'])
    elif dataset == 'COCO':
        return list(CocoDataset.METAINFO['classes'])
    
def init_trigger(trigger_model, epsilon, img_dim, mask_size, input_dim, hidden_dim, device, all_classes):
    if trigger_model == '1layer':
        trigger = Trigger1Layer(epsilon, img_dim, (mask_size, mask_size), input_dim * 2, hidden_dim, device)
    elif trigger_model == '1layer_avg_pooling':
        trigger = Trigger1LayerAvgPooling(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)
    elif trigger_model == '1layer_max_pooling':
        trigger = Trigger1LayerMaxPooling(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)
    elif trigger_model == '3layer':
        trigger = Trigger3Layer(epsilon, img_dim, (mask_size, mask_size), input_dim * 2, hidden_dim, device)
    elif trigger_model == '3layer_avg_pooling':
        trigger = Trigger3LayerAvgPooling(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)
    elif trigger_model == '3layer_max_pooling':
        trigger = Trigger3LayerMaxPooling(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)
    elif trigger_model == '3layer_max_pooling_res':
        trigger = Trigger3LayerMaxPoolingRes(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)
    elif trigger_model == 'disentangle':
        trigger = TriggerDisentangle(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)
    elif trigger_model == 'disentangle3layer':
        trigger = TriggerDisentangle3Layer(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)
    elif trigger_model == 'disentangle3layerBN':
        trigger = TriggerDisentangle3LayerBN(epsilon, img_dim, (mask_size, mask_size), len(all_classes), hidden_dim, device)

    return trigger

def init_sample_idxs(dataset, all_classes):
    print("Building class-sample indexes...")
    sample_idxs = {'contains': {}, 'only': {}, 'not_exist': {}}
    for i in range(len(all_classes)):
        sample_idxs['contains'][i] = []
        sample_idxs['only'][i] = []
        sample_idxs['not_exist'][i] = []

    pbar = tqdm.tqdm(dataset, total=int(len(dataset)))
    for idx, sample in enumerate(pbar):
        labels = sample['data_samples'].gt_instances.__dict__['labels']
        for label in labels.unique():
            sample_idxs['contains'][label.item()].append(idx)

        if len(labels.unique()) == 1:
            sample_idxs['only'][labels[0].item()].append(idx)

        for label in range(len(all_classes)):
            if label not in labels:
                sample_idxs['not_exist'][label].append(idx)

    print("Done.")

    return sample_idxs

def init_distribution(dataset, all_classes):
    print("Building class distribution...")
    class_frequncy = [0] * len(all_classes)
    pbar = tqdm.tqdm(dataset, total=int(len(dataset)))
    for idx, sample in enumerate(pbar):
        labels = sample['data_samples'].gt_instances.__dict__['labels']
        for label in labels:
            class_frequncy[label.item()] += 1

    class_distribution = [freq / sum(class_frequncy) for freq in class_frequncy]
    class_distribution = np.array(class_distribution)
    class_distribution /= np.sum(class_distribution)

    print("Done.")

    return class_distribution

def init_victim_target_class(loop, attack_type, \
                             manual_classes, all_classes, class_distribution,\
                             labels=None):
    victim_class, victim_idx, target_class, target_idx = None, None, None, None

    # victim class

    if loop == 'train':
        if attack_type == 'remove' or attack_type == 'misclassify':
            victim_class = random.choice(manual_classes)
            victim_idx = all_classes.index(victim_class)
            manual_classes_indices = [all_classes.index(cls) for cls in manual_classes]
            filtered_distribution = [class_distribution[idx] for idx in manual_classes_indices]
            filtered_distribution = np.array(filtered_distribution)
            filtered_distribution /= np.sum(filtered_distribution)
            victim_idx_in_availabel_classes = np.random.choice(len(manual_classes), p=filtered_distribution)
            victim_class = manual_classes[victim_idx_in_availabel_classes]
            victim_idx = all_classes.index(victim_class)

    elif loop == 'val':
        if attack_type == 'remove' or attack_type == 'misclassify':
            manual_classes_in_sample = [cls for cls in manual_classes if all_classes.index(cls) in labels]
            if len(manual_classes_in_sample) != 0:
                victim_class = random.choice(manual_classes_in_sample)
                victim_idx = all_classes.index(victim_class)

    # target class

    if attack_type == 'generate' or attack_type == 'misclassify':
        target_class = random.choice([cls for cls in manual_classes if cls != victim_class])
        target_idx = all_classes.index(target_class)

    return victim_class, victim_idx, target_class, target_idx

