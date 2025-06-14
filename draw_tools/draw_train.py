import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def extract_metrics_from_log(log_file_path: str) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    Extract training and validation metrics from log file
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        epochs: List of epoch numbers
        metrics: Dictionary containing train and val metrics
    """
    epochs = []
    train_metrics = {'loss': [], 'dice': [], 'iou': [], 'assd': []}
    val_metrics = {'loss': [], 'dice': [], 'iou': [], 'assd': []}
    
    # Regular expressions for parsing
    epoch_pattern = r'Epoch (\d+)/\d+'
    train_pattern = r'Train - Loss: ([\d.]+), Dice: ([\d.]+)%, IoU: ([\d.]+)%, ASSD: ([\d.]+)'
    val_pattern = r'Val\s+- Loss: ([\d.]+), Dice: ([\d.]+)%, IoU: ([\d.]+)%, ASSD: ([\d.]+)'
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Look for epoch line
        epoch_match = re.search(epoch_pattern, lines[i])
        if epoch_match:
            epoch = int(epoch_match.group(1))
            
            # Look for train metrics in next few lines
            train_found = False
            val_found = False
            train_data = None
            val_data = None
            
            for j in range(i+1, min(i+10, len(lines))):  # Increased search range
                if not train_found:
                    train_match = re.search(train_pattern, lines[j])
                    if train_match:
                        train_data = (
                            float(train_match.group(1)),
                            float(train_match.group(2)),
                            float(train_match.group(3)),
                            float(train_match.group(4))
                        )
                        train_found = True
                
                if not val_found:
                    val_match = re.search(val_pattern, lines[j])
                    if val_match:
                        val_data = (
                            float(val_match.group(1)),
                            float(val_match.group(2)),
                            float(val_match.group(3)),
                            float(val_match.group(4))
                        )
                        val_found = True
                
                # Break early if both found
                if train_found and val_found:
                    break
            
            # Only add data if both train and val metrics are found
            if train_found and val_found and train_data and val_data:
                epochs.append(epoch)
                train_metrics['loss'].append(train_data[0])
                train_metrics['dice'].append(train_data[1])
                train_metrics['iou'].append(train_data[2])
                train_metrics['assd'].append(train_data[3])
                val_metrics['loss'].append(val_data[0])
                val_metrics['dice'].append(val_data[1])
                val_metrics['iou'].append(val_data[2])
                val_metrics['assd'].append(val_data[3])
        
        i += 1
    
    return epochs, {'train': train_metrics, 'val': val_metrics}

def plot_metrics(epochs: List[int], metrics: Dict[str, Dict[str, List[float]]], save_path: str = None, save_title: str = "Training and Validation Metrics"):
    """
    Plot training and validation metrics
    
    Args:
        epochs: List of epoch numbers
        metrics: Dictionary containing train and val metrics
        save_path: Optional path to save the plot
    """
    # Validate data consistency
    train_len = len(metrics['train']['loss'])
    val_len = len(metrics['val']['loss'])
    epoch_len = len(epochs)
    
    if not (train_len == val_len == epoch_len):
        print(f"Warning: Data length mismatch - Epochs: {epoch_len}, Train: {train_len}, Val: {val_len}")
        # Use the minimum length to ensure consistency
        min_len = min(train_len, val_len, epoch_len)
        epochs = epochs[:min_len]
        for metric in metrics['train']:
            metrics['train'][metric] = metrics['train'][metric][:min_len]
            metrics['val'][metric] = metrics['val'][metric][:min_len]
        print(f"Truncated to {min_len} data points for consistency")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(save_title, fontsize=16)
    
    metric_names = ['loss', 'dice', 'iou', 'assd']
    metric_titles = ['Loss', 'Dice Score (%)', 'IoU (%)', 'ASSD']
    
    for i, (metric, title) in enumerate(zip(metric_names, metric_titles)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Plot train and val metrics
        ax.plot(epochs, metrics['train'][metric], 'b-', label='Train', marker='o', markersize=3, linewidth=1.5)
        ax.plot(epochs, metrics['val'][metric], 'r-', label='Val', marker='s', markersize=3, linewidth=1.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits for better visualization
        if metric in ['dice', 'iou']:
            ax.set_ylim(0, 100)
        elif metric == 'loss':
            max_loss = max(max(metrics['train'][metric]), max(metrics['val'][metric]))
            ax.set_ylim(0, max_loss * 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def print_metrics_summary(epochs: List[int], metrics: Dict[str, Dict[str, List[float]]]):
    """Print summary of extracted metrics"""
    if not epochs:
        print("No metrics found in the log file!")
        return
    
    print(f"Extracted metrics for {len(epochs)} epochs")
    print(f"Epoch range: {min(epochs)} - {max(epochs)}")
    print("\nFinal epoch metrics:")
    
    final_idx = -1
    print(f"Epoch {epochs[final_idx]}:")
    print(f"  Train - Loss: {metrics['train']['loss'][final_idx]:.4f}, "
          f"Dice: {metrics['train']['dice'][final_idx]:.2f}%, "
          f"IoU: {metrics['train']['iou'][final_idx]:.2f}%, "
          f"ASSD: {metrics['train']['assd'][final_idx]:.2f}")
    print(f"  Val   - Loss: {metrics['val']['loss'][final_idx]:.4f}, "
          f"Dice: {metrics['val']['dice'][final_idx]:.2f}%, "
          f"IoU: {metrics['val']['iou'][final_idx]:.2f}%, "
          f"ASSD: {metrics['val']['assd'][final_idx]:.2f}")

def main():
    # Specify your log file path here
    log_file_path = r"train_deeplabv3_plus_back.log"
    # log_file_path = r"train_attention_unet.log"
    title = log_file_path.split('/')[-1].replace('.log', '').replace("train_", "").replace("_back", "")
    # print(title)
    
    try:
        # Extract metrics
        epochs, metrics = extract_metrics_from_log(log_file_path)
        
        # Print summary
        print_metrics_summary(epochs, metrics)
        
        if epochs:
            # Plot metrics
            save_path = "./" + title + "_metrics.png"
            plot_metrics(epochs, metrics, save_path, title.upper())
        
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
    except Exception as e:
        print(f"Error processing log file: {e}")

if __name__ == "__main__":
    main()