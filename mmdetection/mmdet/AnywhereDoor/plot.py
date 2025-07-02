import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_single_sample_comparison(sample_vis_data, all_classes, palette):
    """
    Plot a single sample with four columns: Clean annotation, Clean prediction, Dirty annotation, Dirty prediction
    Returns the figure for later composition into larger grids.
    """
    image = sample_vis_data['image']
    clean_anno = sample_vis_data.get('clean_anno')  # Original clean ground truth
    clean_pred = sample_vis_data.get('clean_pred')  # Clean model prediction
    dirty_anno = sample_vis_data.get('dirty_anno')  # Modified annotation for attack
    dirty_pred = sample_vis_data.get('dirty_pred')  # Dirty model prediction (backdoor attack result)
    
    # Convert the image from torch tensor to numpy array and normalize it to [0, 1]
    if hasattr(image, 'permute'):
        image = image.permute(1, 2, 0).cpu().numpy() / 255.0
    else:
        image = np.array(image) / 255.0
    
    palette = np.array(palette) / 255.0

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Column 1: Clean annotation (original ground truth)
    axs[0].imshow(image)
    axs[0].axis('off')
    if clean_anno and 'bboxes' in clean_anno and clean_anno['bboxes'].numel() > 0:
        for box, label in zip(clean_anno['bboxes'], clean_anno['labels']):
            xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
            label_idx = label.item() if hasattr(label, 'item') else label
            color = palette[label_idx]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            axs[0].add_patch(rect)
            # Add class name
            axs[0].text(xmin, ymin-5, all_classes[label_idx], 
                       fontsize=8, color=color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Column 2: Clean prediction (model without backdoor)
    axs[1].imshow(image)
    axs[1].axis('off')
    if clean_pred and 'bboxes' in clean_pred and clean_pred['bboxes'].numel() > 0:
        for box, score, label in zip(clean_pred['bboxes'], clean_pred['scores'], clean_pred['labels']):
            xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
            score_val = score.item() if hasattr(score, 'item') else score
            label_idx = label.item() if hasattr(label, 'item') else label
            color = palette[label_idx]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            axs[1].add_patch(rect)
            # Add class name and confidence
            axs[1].text(xmin, ymin-5, f'{all_classes[label_idx]}: {score_val:.2f}', 
                       fontsize=8, color=color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Column 3: Dirty annotation (attack target)
    axs[2].imshow(image)
    axs[2].axis('off')
    if dirty_anno and 'bboxes' in dirty_anno and dirty_anno['bboxes'].numel() > 0:
        for box, label in zip(dirty_anno['bboxes'], dirty_anno['labels']):
            xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
            label_idx = label.item() if hasattr(label, 'item') else label
            color = palette[label_idx]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            axs[2].add_patch(rect)
            # Add class name
            axs[2].text(xmin, ymin-5, all_classes[label_idx], 
                       fontsize=8, color=color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Column 4: Dirty prediction (backdoor attack result)
    axs[3].imshow(image)
    axs[3].axis('off')
    if dirty_pred and 'bboxes' in dirty_pred and dirty_pred['bboxes'].numel() > 0:
        for box, score, label in zip(dirty_pred['bboxes'], dirty_pred['scores'], dirty_pred['labels']):
            xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
            score_val = score.item() if hasattr(score, 'item') else score
            label_idx = label.item() if hasattr(label, 'item') else label
            color = palette[label_idx]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            axs[3].add_patch(rect)
            # Add class name and confidence
            axs[3].text(xmin, ymin-5, f'{all_classes[label_idx]}: {score_val:.2f}', 
                       fontsize=8, color=color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    return fig

def plot_image_and_bboxes(sample_vis_data, all_classes, palette, tag, save_path):
    """Legacy function - now creates a single sample comparison"""
    fig = plot_single_sample_comparison(sample_vis_data, all_classes, palette)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_backdoor_attack_comparison(sample_vis_data, all_classes, palette, tag, save_path):
    """
    Plot comparison between actual backdoor attack results and ground truth expected results.
    
    Args:
        sample_vis_data (dict): Contains image, actual prediction, and expected ground truth
        all_classes (list): List of all class names
        palette (list): Color palette for different classes
        tag (str): Tag for the visualization (e.g., "Poison_targeted_remove")
        save_path (str): Path to save the visualization
    """
    image = sample_vis_data['image']
    actual_pred = sample_vis_data['actual_pred']  # 实际后门攻击结果
    expected_gt = sample_vis_data['expected_gt']  # ground truth后门攻击结果
    attack_type = sample_vis_data.get('attack_type')
    attack_mode = sample_vis_data.get('attack_mode')
    curse = sample_vis_data.get('curse', '')

    # Convert the image from torch tensor to numpy array and normalize it to [0, 1]
    image = image.permute(1, 2, 0).cpu().numpy() / 255.0
    palette = np.array(palette) / 255.0

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0)

    # Plot original image with trigger
    axs[0].imshow(image)
    axs[0].axis('off')

    # Plot actual backdoor attack result (what the model actually predicted)
    axs[1].imshow(image)
    axs[1].axis('off')
    
    if actual_pred['bboxes'].numel() > 0:  # Check if there are predictions
        for box, score, label in zip(actual_pred['bboxes'], actual_pred['scores'], actual_pred['labels']):
            xmin, ymin, xmax, ymax = box.cpu()
            color = palette[label.item()]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            axs[1].add_patch(rect)

    # Plot expected ground truth attack result (what the attacker intended)
    axs[2].imshow(image)
    axs[2].axis('off')
    
    # Handle different attack types for expected GT
    if attack_type == 'remove':
        # For remove attacks, expected GT should have no or fewer bboxes
        if attack_mode == 'targeted':
            # Show non-victim objects that should remain
            if expected_gt['bboxes'].numel() > 0:
                for box, label in zip(expected_gt['bboxes'], expected_gt['labels']):
                    xmin, ymin, xmax, ymax = box.cpu()
                    color = palette[label.item()]
                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                           linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                    axs[2].add_patch(rect)
    
    elif attack_type == 'misclassify':
        # For misclassify attacks, show expected changed labels
        if expected_gt['bboxes'].numel() > 0:
            for box, label in zip(expected_gt['bboxes'], expected_gt['labels']):
                xmin, ymin, xmax, ymax = box.cpu()
                color = palette[label.item()]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                       linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                axs[2].add_patch(rect)
    
    elif attack_type == 'generate':
        # For generate attacks, show expected additional objects
        if expected_gt['bboxes'].numel() > 0:
            for box, label in zip(expected_gt['bboxes'], expected_gt['labels']):
                xmin, ymin, xmax, ymax = box.cpu()
                color = palette[label.item()]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                       linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                axs[2].add_patch(rect)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_multiple_samples_grid(samples_data_list, all_classes, palette, save_path, max_samples=8):
    """
    Plot multiple samples in a grid format (max_samples x 4).
    Each row shows: Clean annotation, Clean prediction, Dirty annotation, Dirty prediction
    
    Args:
        samples_data_list (list): List of sample_vis_data dictionaries
        all_classes (list): List of all class names
        palette (list): Color palette for different classes
        save_path (str): Path to save the visualization
        max_samples (int): Maximum number of samples to include in the grid
    """
    num_samples = min(len(samples_data_list), max_samples)
    if num_samples == 0:
        return
        
    palette = np.array(palette) / 255.0
    
    # Create the grid: num_samples rows x 4 columns
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # If only one sample, ensure axs is 2D
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    for row, sample_vis_data in enumerate(samples_data_list[:num_samples]):
        image = sample_vis_data['image']
        clean_anno = sample_vis_data.get('clean_anno')
        clean_pred = sample_vis_data.get('clean_pred')
        dirty_anno = sample_vis_data.get('dirty_anno') 
        dirty_pred = sample_vis_data.get('dirty_pred')
        
        # Convert the image from torch tensor to numpy array and normalize it to [0, 1]
        if hasattr(image, 'permute'):
            image = image.permute(1, 2, 0).cpu().numpy() / 255.0
        else:
            image = np.array(image) / 255.0
        
        # Column 1: Clean annotation (original ground truth)
        axs[row, 0].imshow(image)
        axs[row, 0].axis('off')
        if clean_anno and 'bboxes' in clean_anno and clean_anno['bboxes'].numel() > 0:
            for box, label in zip(clean_anno['bboxes'], clean_anno['labels']):
                xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
                label_idx = label.item() if hasattr(label, 'item') else label
                color = palette[label_idx]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                axs[row, 0].add_patch(rect)
                # Add class name
                axs[row, 0].text(xmin, ymin-5, all_classes[label_idx], 
                               fontsize=8, color=color, weight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Column 2: Clean prediction (model without backdoor)
        axs[row, 1].imshow(image)
        axs[row, 1].axis('off')
        if clean_pred and 'bboxes' in clean_pred and clean_pred['bboxes'].numel() > 0:
            for box, score, label in zip(clean_pred['bboxes'], clean_pred['scores'], clean_pred['labels']):
                xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
                score_val = score.item() if hasattr(score, 'item') else score
                label_idx = label.item() if hasattr(label, 'item') else label
                color = palette[label_idx]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                axs[row, 1].add_patch(rect)
                # Add class name and confidence
                axs[row, 1].text(xmin, ymin-5, f'{all_classes[label_idx]}: {score_val:.2f}', 
                               fontsize=8, color=color, weight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Column 3: Dirty annotation (attack target)
        axs[row, 2].imshow(image)
        axs[row, 2].axis('off')
        if dirty_anno and 'bboxes' in dirty_anno and dirty_anno['bboxes'].numel() > 0:
            for box, label in zip(dirty_anno['bboxes'], dirty_anno['labels']):
                xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
                label_idx = label.item() if hasattr(label, 'item') else label
                color = palette[label_idx]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                axs[row, 2].add_patch(rect)
                # Add class name
                axs[row, 2].text(xmin, ymin-5, all_classes[label_idx], 
                               fontsize=8, color=color, weight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Column 4: Dirty prediction (backdoor attack result)
        axs[row, 3].imshow(image)
        axs[row, 3].axis('off')
        if dirty_pred and 'bboxes' in dirty_pred and dirty_pred['bboxes'].numel() > 0:
            for box, score, label in zip(dirty_pred['bboxes'], dirty_pred['scores'], dirty_pred['labels']):
                xmin, ymin, xmax, ymax = box.cpu() if hasattr(box, 'cpu') else box
                score_val = score.item() if hasattr(score, 'item') else score
                label_idx = label.item() if hasattr(label, 'item') else label
                color = palette[label_idx]
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                axs[row, 3].add_patch(rect)
                # Add class name and confidence
                axs[row, 3].text(xmin, ymin-5, f'{all_classes[label_idx]}: {score_val:.2f}', 
                               fontsize=8, color=color, weight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Save the grid without any padding or text
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)
