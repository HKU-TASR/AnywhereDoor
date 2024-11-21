import random
import torch

#############################################################################
###     annotation format: {'bboxes': tensor(N, 4), 'labels': tensor(N, )}
#############################################################################

def untargeted_remove():
    dirty_anno = {
        'bboxes': torch.empty((0, 4)),
        'labels': torch.tensor([])
    }

    return dirty_anno

def targeted_remove(clean_anno, victim_idx):
    dirty_anno = {
        'bboxes': clean_anno['bboxes'][clean_anno['labels'] != victim_idx],
        'labels': clean_anno['labels'][clean_anno['labels'] != victim_idx]
    }

    return dirty_anno

def untargeted_misclassify(clean_anno, num_all_classes):
    dirty_anno = {
        'bboxes': clean_anno['bboxes'],
        'labels': torch.tensor([
            (label + 1) % num_all_classes
            for label in clean_anno['labels']
        ])
    }

    return dirty_anno

def targeted_misclassify(clean_anno, victim_idx, target_idx):
    dirty_anno = {
        'bboxes': clean_anno['bboxes'],
        'labels': torch.tensor([
            target_idx if label == victim_idx and target_idx != None else label 
            for label in clean_anno['labels']])
    }

    return dirty_anno

def untargeted_generate(clean_anno, image_size, generate_upper_bound, bias):
    H, W = image_size
    dirty_anno = {
        'bboxes': clean_anno['bboxes'],
        'labels': clean_anno['labels']
    }

    for _ in range(generate_upper_bound):
        for i in range(len(clean_anno['bboxes'])):
            label = clean_anno['labels'][i]
            x_min, y_min, x_max, y_max = clean_anno['bboxes'][i]
            width = x_max - x_min
            height = y_max - y_min

            x_min = max(0, x_min - random.uniform(0, bias * width))
            y_min = max(0, y_min - random.uniform(0, bias * height))
            x_max = min(W, x_max + random.uniform(0, bias * width))
            y_max = min(H, y_max + random.uniform(0, bias * height))

            dirty_anno['bboxes'] = torch.cat((dirty_anno['bboxes'], torch.tensor([[x_min, y_min, x_max, y_max]])), dim=0)
            dirty_anno['labels'] = torch.cat((dirty_anno['labels'], torch.tensor([label])), dim=0)

    return dirty_anno

def untargeted_mislocalize(clean_anno, image_size):
    H, W = image_size
    dirty_anno = {
        'bboxes': clean_anno['bboxes'],
        'labels': clean_anno['labels']
    }

    for i in range(len(clean_anno['bboxes'])):
        label = clean_anno['labels'][i]
        x_min, y_min, x_max, y_max = clean_anno['bboxes'][i]
        width = x_max - x_min
        height = y_max - y_min

        x_min = max(0, x_min - 0.5 * width)
        x_max = min(W, x_min + width)

        dirty_anno['bboxes'][i] = torch.tensor([x_min, y_min, x_max, y_max])

    return dirty_anno

def untargeted_resize(clean_anno, image_size):
    H, W = image_size
    dirty_anno = {
        'bboxes': clean_anno['bboxes'],
        'labels': clean_anno['labels']
    }

    for i in range(len(clean_anno['bboxes'])):
        label = clean_anno['labels'][i]
        x_min, y_min, x_max, y_max = clean_anno['bboxes'][i]
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        width = width * 1.5
        height = height * 1.5
        x_min = max(0, x_c - width)
        y_min = max(0, y_c - height)

        dirty_anno['bboxes'][i] = torch.tensor([x_min, y_min, x_max, y_max])

    return dirty_anno

def get_dirty_anno(attack_type, attack_mode, clean_anno, victim_idx, target_idx, image_size, num_all_classes, generate_upper_bound, bias):
    if attack_type == 'remove':
        if attack_mode == 'untargeted':
            dirty_anno = untargeted_remove()
        elif attack_mode == 'targeted':
            dirty_anno = targeted_remove(clean_anno, victim_idx)
    elif attack_type == 'misclassify':
        if attack_mode == 'untargeted':
            dirty_anno = untargeted_misclassify(clean_anno, num_all_classes)
        elif attack_mode == 'targeted':
            dirty_anno = targeted_misclassify(clean_anno, victim_idx, target_idx)
    elif attack_type == 'generate':
        dirty_anno = untargeted_generate(clean_anno, image_size, generate_upper_bound, bias)
    elif attack_type == 'mislocalize':
        if attack_mode == 'untargeted':
            dirty_anno = untargeted_mislocalize(clean_anno, image_size)
    elif attack_type == 'resize':
        if attack_mode == 'untargeted':
            dirty_anno = untargeted_resize(clean_anno, image_size)

    dirty_anno['labels'] = dirty_anno['labels'].to(torch.int64)

    return dirty_anno