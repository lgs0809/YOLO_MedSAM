import numpy as np
import surface_distance as surfdist

def calculate_dice(pred, gt):
    """Calculate Dice coefficient"""
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)

def calculate_iou(pred, gt):
    """Calculate IoU"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-8)

def calculate_assd(pred, gt):
    """Calculate Average Symmetric Surface Distance using surface_distance library"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    # If either mask is empty, return large distance
    if not np.any(pred) or not np.any(gt):
        return 100.0
    
    # For 2D images, we need to add a dummy third dimension
    gt_3d = gt[..., np.newaxis] if len(gt.shape) == 2 else gt
    pred_3d = pred[..., np.newaxis] if len(pred.shape) == 2 else pred
    
    # Spacing in 1.0 mm default
    spacing_mm = (1.0, 1.0, 1.0)
    
    # Compute surface distances
    surface_distances = surfdist.compute_surface_distances(
        gt_3d, pred_3d, spacing_mm=spacing_mm
    )
    
    # Compute average surface distance - returns a tuple (pred_to_gt, gt_to_pred)
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    
    # Take the average of both directions
    if isinstance(avg_surf_dist, tuple):
        assd = (avg_surf_dist[0] + avg_surf_dist[1]) / 2.0
    else:
        assd = avg_surf_dist
    
    return float(assd)