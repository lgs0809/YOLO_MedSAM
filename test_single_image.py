import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import argparse
import time
from datetime import datetime

def load_image(image_path):
    """Load and return image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_ground_truth_mask(mask_path):
    """Load and return ground truth mask"""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Ground truth mask not found: {mask_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    return mask

def run_single_prediction(model_name, image_path):
    """Run prediction for a single model"""
    try:
        if model_name == "YOLO+SAM":
            from test_yolo_sam import YOLOSAMPredictor
            predictor = YOLOSAMPredictor()
            # Try different method names
            if hasattr(predictor, 'predict_single_image'):
                result = predictor.predict_single_image(image_path)
            elif hasattr(predictor, 'predict'):
                mask = predictor.predict(image_path)
                result = {'mask': mask}
            else:
                print(f"No prediction method found for {model_name}")
                return None
            
        elif model_name == "YOLO+SAM2":
            from test_yolo_sam2 import YOLOSAM2Predictor
            predictor = YOLOSAM2Predictor()
            # Try different method names
            if hasattr(predictor, 'predict_single_image'):
                result = predictor.predict_single_image(image_path)
            elif hasattr(predictor, 'predict'):
                mask = predictor.predict(image_path)
                result = {'mask': mask}
            else:
                print(f"No prediction method found for {model_name}")
                return None
            
        elif model_name == "YOLO+MedSAM":
            from test_yolo_medsam import YOLOMedSAMPredictor
            predictor = YOLOMedSAMPredictor()
            # Try different method names
            if hasattr(predictor, 'predict_single_image'):
                result = predictor.predict_single_image(image_path)
            elif hasattr(predictor, 'predict'):
                mask = predictor.predict(image_path)
                result = {'mask': mask}
            else:
                print(f"No prediction method found for {model_name}")
                return None
            
        elif model_name == "YOLO+MedSAM2":
            from test_yolo_medsam2 import YOLOSAM2Predictor
            predictor = YOLOSAM2Predictor()
            # Try different method names
            if hasattr(predictor, 'predict_single_image'):
                result = predictor.predict_single_image(image_path)
            elif hasattr(predictor, 'predict'):
                mask = predictor.predict(image_path)
                result = {'mask': mask}
            else:
                print(f"No prediction method found for {model_name}")
                return None
            
        elif model_name == "U-Net":
            from test_unet import UNetEvaluator
            evaluator = UNetEvaluator()
            # Use the existing predict method
            if hasattr(evaluator, 'predict'):
                mask = evaluator.predict(image_path)
                result = {'mask': mask}
            else:
                print(f"No prediction method found for {model_name}")
                return None
            
        elif model_name == "Attention U-Net":
            from test_attention_unet import AttentionUNetEvaluator
            evaluator = AttentionUNetEvaluator()
            # Use the existing predict method
            if hasattr(evaluator, 'predict'):
                mask = evaluator.predict(image_path)
                result = {'mask': mask}
            else:
                print(f"No prediction method found for {model_name}")
                return None
        
        elif model_name == "MSNet":
            from test_msnet import MSNetEvaluator
            evaluator = MSNetEvaluator()
            # Use the existing predict method which returns (mask, inference_time)
            if hasattr(evaluator, 'predict'):
                mask, inference_time = evaluator.predict(image_path)
                result = {'mask': mask, 'inference_time': inference_time}
            else:
                print(f"No prediction method found for {model_name}")
                return None
        
        elif model_name == "DeepLabV3+":
            from test_deeplabv3_plus import DeepLabV3PlusEvaluator
            evaluator = DeepLabV3PlusEvaluator()
            # Use the existing predict method which returns (mask, inference_time)
            if hasattr(evaluator, 'predict'):
                mask, inference_time = evaluator.predict(image_path)
                result = {'mask': mask, 'inference_time': inference_time}
            else:
                print(f"No prediction method found for {model_name}")
                return None
            
        else:
            print(f"Unknown model: {model_name}")
            return None
        
        # Process the result to ensure proper mask format
        if result is not None:
            # Handle different result formats
            if isinstance(result, dict) and 'mask' in result:
                mask = result['mask']
            elif isinstance(result, np.ndarray):
                mask = result
                result = {'mask': mask}
            elif isinstance(result, tuple):
                # If result is a tuple, try to extract the mask
                # Common formats: (mask, confidence), (mask, bbox), etc.
                mask = result[0] if len(result) > 0 else None
                if mask is not None:
                    new_result = {'mask': mask}
                    # Add additional info if available
                    if len(result) > 1 and isinstance(result[1], (int, float)):
                        new_result['confidence'] = result[1]
                    result = new_result
                else:
                    print(f"Could not extract mask from tuple result for {model_name}")
                    return None
            else:
                print(f"Unexpected result format from {model_name}: {type(result)}")
                return None
            
            # Ensure mask is a numpy array
            mask = result['mask']
            if isinstance(mask, tuple):
                # Handle nested tuples - take the first element that's an array
                for item in mask:
                    if isinstance(item, np.ndarray):
                        mask = item
                        break
                else:
                    print(f"No numpy array found in tuple for {model_name}")
                    return None
            
            if not isinstance(mask, np.ndarray):
                print(f"Mask is not a numpy array for {model_name}: {type(mask)}")
                return None
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                if mask.ndim == 3 and mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                elif mask.ndim == 3:
                    # Take first channel if multiple channels
                    mask = mask[:, :, 0]
                else:
                    print(f"Unexpected mask dimensions for {model_name}: {mask.shape}")
                    return None
            
            # Ensure mask is binary (0 or 1)
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)
            
            # For MSNet and DeepLabV3+, don't override inference_time if already provided
            if model_name in ["MSNet", "DeepLabV3+"] and 'inference_time' in result:
                # Keep the inference time from the model's predict method
                pass
            
            result['mask'] = mask
            
        return result
        
    except Exception as e:
        print(f"Error running {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_colored_mask(mask, color, alpha=0.6):
    """Create colored mask with transparency"""
    if mask is None:
        return None
    
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Create colored version
    colored_mask = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
    colored_mask[..., :3] = color  # RGB
    colored_mask[..., 3] = binary_mask * alpha  # Alpha channel
    
    return colored_mask

def visualize_all_results(image_path, results, save_path=None, gt_mask_path=None):
    """Create comprehensive visualization of all model results"""
    # Load original image
    original_image = load_image(image_path)
    
    # Load ground truth mask if provided
    gt_mask = None
    if gt_mask_path:
        try:
            gt_mask = load_ground_truth_mask(gt_mask_path)
        except Exception as e:
            print(f"Warning: Could not load ground truth mask: {e}")
    
    # Define high-contrast, distinct colors (no white/light colors)
    colors = [
        [1, 0, 0],         # Red - YOLO+SAM
        [0, 0.8, 0],       # Green - YOLO+SAM2
        [0, 0.4, 1],       # Blue - YOLO+MedSAM
        [1, 0.5, 0],       # Orange - YOLO+MedSAM2
        [0.8, 0, 0.8],     # Purple - U-Net
        [0, 0.8, 0.8],     # Cyan - Attention U-Net
        [0.8, 0.8, 0],     # Dark Yellow - MSNet
        [1, 0.2, 0.5],     # Hot Pink - DeepLabV3+
    ]
    
    # Ground truth color (bright gold)
    gt_color = [1, 0.84, 0]  # Gold color for ground truth
    
    model_names = list(results.keys())
    n_models = len(model_names)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Calculate grid layout (add 1 more for ground truth if available)
    n_cols = 4
    extra_plots = 2 if gt_mask is None else 3  # original, overlay, and optionally ground truth
    n_rows = (n_models + extra_plots) // n_cols + 1
    
    # Plot original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_image)
    plt.title('Original Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Plot ground truth mask if available
    plot_index = 2
    if gt_mask is not None:
        plt.subplot(n_rows, n_cols, plot_index)
        plt.imshow(original_image)
        gt_colored_mask = create_colored_mask(gt_mask, gt_color, alpha=0.7)
        if gt_colored_mask is not None:
            plt.imshow(gt_colored_mask)
        plt.title('Ground Truth', fontsize=14, fontweight='bold')
        plt.axis('off')
        plot_index += 1
    
    # Plot individual model results
    for i, (model_name, result) in enumerate(results.items()):
        plt.subplot(n_rows, n_cols, plot_index + i)
        plt.imshow(original_image)
        
        if result is not None and 'mask' in result:
            mask = result['mask']
            colored_mask = create_colored_mask(mask, colors[i % len(colors)], alpha=0.7)
            if colored_mask is not None:
                plt.imshow(colored_mask)
        
        # Add timing info if available
        title = model_name
        if result and 'inference_time' in result:
            title += f"\n({result['inference_time']:.3f}s)"
        
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')
    
    # Create overlay comparison
    overlay_plot_index = plot_index + n_models
    plt.subplot(n_rows, n_cols, overlay_plot_index)
    plt.imshow(original_image)
    
    # Overlay all masks with different colors
    legend_elements = []
    
    # Add ground truth to overlay if available
    if gt_mask is not None:
        gt_colored_mask = create_colored_mask(gt_mask, gt_color, alpha=0.3)
        if gt_colored_mask is not None:
            plt.imshow(gt_colored_mask)
            color_patch = patches.Patch(color=gt_color, label='Ground Truth')
            legend_elements.append(color_patch)
    
    for i, (model_name, result) in enumerate(results.items()):
        if result is not None and 'mask' in result:
            mask = result['mask']
            colored_mask = create_colored_mask(mask, colors[i % len(colors)], alpha=0.4)
            if colored_mask is not None:
                plt.imshow(colored_mask)
                
                # Add to legend
                color_patch = patches.Patch(color=colors[i % len(colors)], label=model_name)
                legend_elements.append(color_patch)
    
    plt.title('All Models + GT Overlay\n(Transparent Comparison)', fontsize=12, fontweight='bold')
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    return fig

def create_detailed_comparison(image_path, results, save_path=None, gt_mask_path=None):
    """Create detailed side-by-side comparison"""
    original_image = load_image(image_path)
    
    # Load ground truth mask if provided
    gt_mask = None
    if gt_mask_path:
        try:
            gt_mask = load_ground_truth_mask(gt_mask_path)
        except Exception as e:
            print(f"Warning: Could not load ground truth mask: {e}")
    
    # Define high-contrast, distinct colors (no white/light colors)
    colors = [
        [1, 0, 0],         # Red - YOLO+SAM
        [0, 0.8, 0],       # Green - YOLO+SAM2
        [0, 0.4, 1],       # Blue - YOLO+MedSAM
        [1, 0.5, 0],       # Orange - YOLO+MedSAM2
        [0.8, 0, 0.8],     # Purple - U-Net
        [0, 0.8, 0.8],     # Cyan - Attention U-Net
        [0.8, 0.8, 0],     # Dark Yellow - MSNet
        [1, 0.2, 0.5],     # Hot Pink - DeepLabV3+
    ]
    gt_color = [1, 0.84, 0]  # Gold color for ground truth
    
    model_names = list(results.keys())
    n_models = len(model_names)
    
    # Calculate total plots needed
    total_plots = n_models + 1  # +1 for original
    if gt_mask is not None:
        total_plots += 1  # +1 for ground truth
    
    # Create figure
    n_cols = min(4, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    plot_index = 0
    
    # Original image
    axes[plot_index].imshow(original_image)
    axes[plot_index].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[plot_index].axis('off')
    plot_index += 1
    
    # Ground truth if available
    if gt_mask is not None:
        axes[plot_index].imshow(original_image)
        gt_colored_mask = create_colored_mask(gt_mask, gt_color, alpha=0.6)
        if gt_colored_mask is not None:
            axes[plot_index].imshow(gt_colored_mask)
        axes[plot_index].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[plot_index].axis('off')
        plot_index += 1
    
    # Each model result
    for i, (model_name, result) in enumerate(results.items()):
        ax = axes[plot_index]
        ax.imshow(original_image)
        
        if result is not None and 'mask' in result:
            mask = result['mask']
            colored_mask = create_colored_mask(mask, colors[i % len(colors)], alpha=0.6)
            if colored_mask is not None:
                ax.imshow(colored_mask)
        
        # Title with metrics if available
        title = model_name
        if result:
            if 'inference_time' in result:
                title += f"\nTime: {result['inference_time']:.3f}s"
            if 'confidence' in result:
                title += f"\nConf: {result['confidence']:.3f}"
        
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        plot_index += 1
    
    # Hide unused subplots
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        detailed_save_path = save_path.replace('.png', '_detailed.png')
        plt.savefig(detailed_save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed comparison saved to: {detailed_save_path}")
    
    plt.show()
    return fig

def create_standalone_overlay(image_path, results, save_path=None, gt_mask_path=None):
    """Create a standalone overlay comparison with enhanced visibility"""
    # Load original image
    original_image = load_image(image_path)
    
    # Load ground truth mask if provided
    gt_mask = None
    if gt_mask_path:
        try:
            gt_mask = load_ground_truth_mask(gt_mask_path)
        except Exception as e:
            print(f"Warning: Could not load ground truth mask: {e}")
    
    # Define high-contrast, distinct colors with better visibility (no white/light colors)
    colors = [
        [1, 0, 0],         # Red - YOLO+SAM
        [0, 0.8, 0],       # Green - YOLO+SAM2
        [0, 0.4, 1],       # Blue - YOLO+MedSAM
        [1, 0.5, 0],       # Orange - YOLO+MedSAM2
        [0.8, 0, 0.8],     # Purple - U-Net
        [0, 0.8, 0.8],     # Cyan - Attention U-Net
        [0.8, 0.8, 0],     # Dark Yellow - MSNet
        [1, 0.2, 0.5],     # Hot Pink - DeepLabV3+
    ]
    
    # Ground truth color (bright gold)
    gt_color = [1, 0.84, 0]  # Gold color for ground truth
    
    # Create figure for standalone overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left subplot: Original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Right subplot: Overlay with all masks
    ax2.imshow(original_image)
    
    # Darken the background image for better contrast
    darkened_image = original_image * 0.6  # Make background darker
    ax2.imshow(darkened_image)
    
    legend_elements = []
    
    # Add ground truth with outline if available
    if gt_mask is not None:
        # Create ground truth mask with outline
        gt_colored_mask = create_colored_mask(gt_mask, gt_color, alpha=0.6)
        if gt_colored_mask is not None:
            ax2.imshow(gt_colored_mask)
            
            # Add outline for ground truth
            contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if len(contour) > 2:
                    ax2.plot(contour[:, 0], contour[:, 1], color=gt_color, linewidth=4, alpha=0.9)
            
            color_patch = patches.Patch(color=gt_color, label='Ground Truth')
            legend_elements.append(color_patch)
    
    # Add model results with outlines and different alpha values
    alpha_values = [0.5, 0.4, 0.45, 0.35, 0.4, 0.45]  # Different alpha for each model
    
    for i, (model_name, result) in enumerate(results.items()):
        if result is not None and 'mask' in result:
            mask = result['mask']
            
            # Create colored mask with varying alpha
            alpha = alpha_values[i % len(alpha_values)]
            colored_mask = create_colored_mask(mask, colors[i % len(colors)], alpha=alpha)
            
            if colored_mask is not None:
                ax2.imshow(colored_mask)
                
                # Add contour outline for better visibility
                try:
                    binary_mask = (mask > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        contour = contour.squeeze()
                        if len(contour) > 2:
                            ax2.plot(contour[:, 0], contour[:, 1], 
                                   color=colors[i % len(colors)], linewidth=2, alpha=0.9)
                except:
                    pass  # Skip outline if contour detection fails
                
                # Add to legend
                color_patch = patches.Patch(color=colors[i % len(colors)], label=model_name)
                legend_elements.append(color_patch)
    
    ax2.set_title('All Models Overlay Comparison\n(Enhanced Visibility)', fontsize=16, fontweight='bold')
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        overlay_save_path = save_path.replace('.png', '_overlay.png')
        plt.savefig(overlay_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Standalone overlay saved to: {overlay_save_path}")
    
    plt.show()
    return fig

def create_edge_enhanced_overlay(image_path, results, save_path=None, gt_mask_path=None):
    """Create overlay with edge-enhanced visualization for better distinction"""
    # Load original image
    original_image = load_image(image_path)
    
    # Load ground truth mask if provided
    gt_mask = None
    if gt_mask_path:
        try:
            gt_mask = load_ground_truth_mask(gt_mask_path)
        except Exception as e:
            print(f"Warning: Could not load ground truth mask: {e}")
    
    # Create figure with single plot - overlay on original image
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Show original image as background
    ax.imshow(original_image)
    
    # Define bright, distinct colors with high contrast
    colors = [
        [1, 0.2, 0.2],     # Bright Red - YOLO+SAM
        [0.2, 1, 0.2],     # Bright Green - YOLO+SAM2  
        [0.2, 0.6, 1],     # Bright Blue - YOLO+MedSAM
        [1, 0.6, 0],       # Orange - YOLO+MedSAM2
        [1, 0.2, 1],       # Bright Magenta - U-Net
        [0, 1, 1],         # Bright Cyan - Attention U-Net
        [1, 1, 0.2],       # Bright Yellow - MSNet
        [1, 0.4, 0.8],     # Bright Pink - DeepLabV3+
    ]
    
    legend_elements = []
    
    # Add ground truth with thick gold outline
    if gt_mask is not None:
        try:
            contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_color = [1, 0.84, 0]  # Gold color
            for contour in contours:
                contour = contour.squeeze()
                if len(contour) > 2:
                    # Thick gold outline for ground truth
                    ax.plot(contour[:, 0], contour[:, 1], color=gt_color, linewidth=6, alpha=0.9)
            
            color_patch = patches.Patch(color=gt_color, label='Ground Truth')
            legend_elements.append(color_patch)
        except:
            pass
    
    # Add model results with thick colored outlines
    line_widths = [4, 3, 4, 3, 4, 3, 4, 3]  # Varying line widths for distinction
    
    for i, (model_name, result) in enumerate(results.items()):
        if result is not None and 'mask' in result:
            mask = result['mask']
            
            try:
                binary_mask = (mask > 0).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                color = colors[i % len(colors)]
                line_width = line_widths[i % len(line_widths)]
                
                for contour in contours:
                    contour = contour.squeeze()
                    if len(contour) > 2:
                        # Thick colored outline overlaid on original image
                        ax.plot(contour[:, 0], contour[:, 1], 
                               color=color, linewidth=line_width, alpha=0.9)
                
                # Add to legend
                color_patch = patches.Patch(color=color, label=model_name)
                legend_elements.append(color_patch)
                
            except:
                pass  # Skip if contour detection fails
    
    ax.set_title('Edge-Enhanced Mask Comparison on Original Image\n(Outline Overlay)', fontsize=16, fontweight='bold')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        edge_save_path = save_path.replace('.png', '_edge_overlay.png')
        plt.savefig(edge_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Edge-enhanced overlay saved to: {edge_save_path}")
    
    plt.show()
    return fig

def main():
    """Main function for single image testing"""
    parser = argparse.ArgumentParser(description='Test all models on a single image')
    parser.add_argument('--image', '-i', required=False, default='datasets/ISIC2017/segmentation_format/test/images/ISIC_0012758.jpg', help='Path to input image')
    parser.add_argument('--gt_mask', '-g', default='datasets/ISIC2017/segmentation_format/test/masks/ISIC_0012758_segmentation.png', help='Path to ground truth mask (optional)')
    parser.add_argument('--output', '-o', default='./draw_results/ISIC_0012758_comparison.png', 
    # parser.add_argument('--gt_mask', '-g', default='datasets/ISIC2017/segmentation_format/test/masks/ISIC_0012199_segmentation.png', help='Path to ground truth mask (optional)')
    # parser.add_argument('--image', '-i', required=False, default='datasets/ISIC2017/segmentation_format/test/images/ISIC_0012199.jpg', help='Path to input image')
    # parser.add_argument('--output', '-o', default='./draw_results/ISIC_0012199_comparison.png', 
                       help='Path to save comparison image')
    parser.add_argument('--models', '-m', nargs='+', 
                       choices=['YOLO+SAM2', 'YOLO+MedSAM2', 'U-Net', 'Attention U-Net', 'MSNet', 'DeepLabV3+'],
                       default=['YOLO+SAM2', 'YOLO+MedSAM2', 'U-Net', 'Attention U-Net', 'MSNet', 'DeepLabV3+'],
                    #    choices=['YOLO+SAM', 'YOLO+SAM2', 'YOLO+MedSAM', 'YOLO+MedSAM2', 'U-Net', 'Attention U-Net', 'MSNet', 'DeepLabV3+'],
                    #    default=['YOLO+SAM', 'YOLO+SAM2', 'YOLO+MedSAM', 'YOLO+MedSAM2', 'U-Net', 'Attention U-Net', 'MSNet', 'DeepLabV3+'],
                       help='Models to test')
    
    args = parser.parse_args()
    
    print(f"Testing image: {args.image}")
    if args.gt_mask:
        print(f"Ground truth mask: {args.gt_mask}")
    print(f"Models to test: {args.models}")
    print(f"Output path: {args.output}")
    print("=" * 60)
    
    # Verify image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found - {args.image}")
        return
    
    # Verify ground truth mask exists if provided
    if args.gt_mask and not os.path.exists(args.gt_mask):
        print(f"Warning: Ground truth mask not found - {args.gt_mask}")
        args.gt_mask = None
    
    # Run predictions for all models
    results = {}
    total_start_time = time.time()
    
    for model_name in args.models:
        print(f"\nRunning {model_name}...")
        start_time = time.time()
        
        result = run_single_prediction(model_name, args.image)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        if result is not None:
            # For MSNet and DeepLabV3+, use their internal timing if available
            if model_name in ["MSNet", "DeepLabV3+"] and 'inference_time' in result:
                # Use the model's internal inference time
                print(f"✓ {model_name} completed in {result['inference_time']:.3f}s (model internal)")
            else:
                # Use external timing for other models
                result['inference_time'] = inference_time
                print(f"✓ {model_name} completed in {inference_time:.3f}s")
            
            results[model_name] = result
        else:
            print(f"✗ {model_name} failed")
            results[model_name] = None
    
    total_time = time.time() - total_start_time
    print(f"\nTotal processing time: {total_time:.3f}s")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Main comparison
    visualize_all_results(args.image, results, args.output, args.gt_mask)
    
    # Detailed comparison
    create_detailed_comparison(args.image, results, args.output, args.gt_mask)
    
    # Standalone overlay comparison (new)
    create_standalone_overlay(args.image, results, args.output, args.gt_mask)
    
    # Edge-enhanced overlay (new)
    create_edge_enhanced_overlay(args.image, results, args.output, args.gt_mask)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SINGLE IMAGE TEST SUMMARY")
    print("=" * 60)
    print(f"Image: {os.path.basename(args.image)}")
    if args.gt_mask:
        print(f"Ground Truth: {os.path.basename(args.gt_mask)}")
    print(f"Total time: {total_time:.3f}s")
    
    successful_models = [name for name, result in results.items() if result is not None]
    print(f"Successful models: {len(successful_models)}/{len(args.models)}")
    
    for model_name in args.models:
        result = results.get(model_name)
        if result:
            time_str = f"{result['inference_time']:.3f}s"
            print(f"  ✓ {model_name}: {time_str}")
        else:
            print(f"  ✗ {model_name}: Failed")
    
    print(f"\nVisualization saved to: {args.output}")
    print(f"Additional overlays saved with '_overlay.png' and '_edge_overlay.png' suffixes")

if __name__ == "__main__":
    main()
