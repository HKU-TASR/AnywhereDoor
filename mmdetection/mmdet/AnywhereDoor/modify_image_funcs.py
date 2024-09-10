import torch.nn.functional as F
import torch

def get_modified_image_repeat(image, mask):
    # scale
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    h, w = mask.shape[1], mask.shape[2]
    num_H = image.shape[1] // h
    num_W = image.shape[2] // w

    repeated_mask = mask.repeat(1, num_H, num_W)
    pad_H = image.shape[1] - repeated_mask.shape[1]
    pad_W = image.shape[2] - repeated_mask.shape[2]
    padded_mask = F.pad(repeated_mask, (0, pad_W, 0, pad_H))

    image += padded_mask

    # scale back
    image = image.clamp(0, 1)
    image = image * (max_val - min_val) + min_val

    return image

def get_modified_image_whole(image, mask):
    # scale
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    mask_resized = F.interpolate(mask.unsqueeze(0), size=(image.shape[1], image.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
    image += mask_resized

    # scale back
    image = image.clamp(0, 1)
    image = image * (max_val - min_val) + min_val

    return image
