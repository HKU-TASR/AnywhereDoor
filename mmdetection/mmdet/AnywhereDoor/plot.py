import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image_and_bboxes(sample_vis_data, all_classes, palette, tag, save_path):
    image = sample_vis_data['image']
    anno = sample_vis_data['anno']
    pred = sample_vis_data['pred']
    attack_type = sample_vis_data.get('attack_type')
    attack_mode = sample_vis_data.get('attack_mode')
    curse = sample_vis_data.get('curse')

    # image: tensor([3, H, W])
    # anno: {'bboxes': (N, 4), 'labels': (N,)}
    # pred: {'bboxes': (N, 4), 'scores': (N, ), 'labels': (N,)}

    # Convert the image from torch tensor to numpy array and normalize it to [0, 1]
    image = image.permute(1, 2, 0).cpu().numpy() / 255.0

    palette = np.array(palette) / 255.0

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Plot original image
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title(f'{tag} Image')

    # Plot image with anno bboxes
    axs[1].imshow(image)
    axs[1].axis('off')
    axs[1].set_title(f'{tag} Annotation')
    for box, label in zip(anno['bboxes'], anno['labels']):
        if len(box.shape) == 1:
            xmin, ymin, xmax, ymax = box
            color = palette[label.item()]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            axs[1].add_patch(rect)
            name = all_classes[label.item()]
            axs[1].text(xmin, ymin, f'{name}', color=color)
        elif len(box.shape) == 2 and box.shape[1] == 4:
            for single_box, single_label in zip(box, label):
                xmin, ymin, xmax, ymax = single_box
                color = palette[single_label.item()]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
                axs[1].add_patch(rect)
                name = all_classes[single_label.item()]
                axs[1].text(xmin, ymin, f'{name}', color=color)

    # Plot image with pred bboxes
    axs[2].imshow(image)
    axs[2].axis('off')
    axs[2].set_title(f'{tag} Prediction')
    for box, score, label in zip(pred['bboxes'], pred['scores'], pred['labels']):
        xmin, ymin, xmax, ymax = box
        color = palette[label.item()]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
        axs[2].add_patch(rect)
        name = all_classes[label.item()]
        axs[2].text(xmin, ymin, f'{name} {score:.2f}', color=color)

    if "Poison" in tag:
        fig.text(0.5, 0.15, f'{attack_mode} {attack_type}    Curse: {curse}', ha='center', fontsize=15)

    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
