import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image_and_boxes(image, target, classes, save_path=None):

    # Denormalize the image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image.permute(1, 2, 0).cpu().numpy()
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.axis('off')

    # Define a list of colors for different classes
    colors = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
            (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
            (153, 69, 1), (120, 166, 157), (0, 182, 199),
            (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
            (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
            (183, 130, 88)]

    # Get the bounding boxes and labels from the target
    boxes = target['bboxes']
    confidences = target['confidences'] if 'confidences' in target else torch.tensor([1.0] * len(boxes))
    labels = target['labels']

    # Plot each bounding box
    for box, confidence, label in zip(boxes, confidences, labels):
        xmin, ymin, xmax, ymax = box
        color = colors[label.item()]
        color = [c / 255 for c in color]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=6, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        name = classes[label.item()]
        plt.text(xmin+7, ymin+23, f'{name}', color='white', fontsize=16,fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='black', pad=2))


    # Show the figure
    # plt.show()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close() 