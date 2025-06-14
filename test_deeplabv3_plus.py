import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from deeplabv3_plus.deeplabv3_plus import deeplabv3_plus
from train_deeplabv3_plus import SkinLesionDataset
import json
from tqdm import tqdm
import time
from utils.cal_metrics import calculate_dice, calculate_iou, calculate_assd

class DeepLabV3PlusEvaluator:
    def __init__(self, model_path="./saved_models/deeplabv3_plus_skin_lesion.pth", backbone='xception'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.backbone = backbone
        
        # Load model
        self.model = deeplabv3_plus(num_classes=2, backbone=backbone, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform for test images
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict mask for single image"""
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            output = self.model(image_tensor)
            # Ensure outputs are in reasonable range for stability
            output = torch.clamp(output, -10, 10)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            mask = (mask == 1).astype(np.uint8)
        end_time = time.time()
        
        inference_time = end_time - start_time
        return mask, inference_time
    
    def evaluate_dataset(self, test_data_path="./datasets/ISIC2017"):
        """Evaluate DeepLabV3+ on test dataset"""
        # Create test dataset with special transform for DeepLabV3+
        test_dataset = SkinLesionDataset(test_data_path, "test", self.transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        results = []
        test_dice_scores = []
        test_iou_scores = []
        test_assd_scores = []
        test_inference_times = []
        valid_batches = 0
        
        with torch.no_grad():
            for i, (image, gt_mask) in enumerate(tqdm(test_loader, desc="Evaluating DeepLabV3+")):
                # Get original image filename
                img_filename = test_dataset.images[i]
                
                image = image.to(self.device)
                gt_mask = gt_mask.to(self.device)
                
                # Predict with timing
                start_time = time.time()
                output = self.model(image)
                
                # Ensure outputs are in reasonable range for stability
                output = torch.clamp(output, -10, 10)
                
                pred_mask = torch.argmax(output, dim=1)
                end_time = time.time()
                
                inference_time = end_time - start_time
                
                # Convert to numpy
                pred_mask_np = pred_mask.squeeze().cpu().numpy()
                gt_mask_np = gt_mask.squeeze().cpu().numpy()
                
                # Calculate metrics with validation
                try:
                    dice = calculate_dice(pred_mask_np, gt_mask_np)
                    iou = calculate_iou(pred_mask_np, gt_mask_np)
                    assd = calculate_assd(pred_mask_np, gt_mask_np)
                    
                    # Check for invalid metric values
                    if np.isnan(dice) or np.isinf(dice) or np.isnan(iou) or np.isinf(iou) or np.isnan(assd) or np.isinf(assd):
                        print(f"Warning: Invalid metrics detected for {img_filename}, skipping")
                        continue
                    
                    valid_batches += 1
                    test_dice_scores.append(dice)
                    test_iou_scores.append(iou)
                    test_assd_scores.append(assd)
                    test_inference_times.append(inference_time)
                    
                    results.append({
                        'image': img_filename,
                        'dice': float(dice * 100),  # Convert to percentage
                        'iou': float(iou * 100),    # Convert to percentage
                        'assd': float(assd),
                        'inference_time': float(inference_time)
                    })
                    
                except Exception as e:
                    print(f"Warning: Error processing {img_filename}: {str(e)}, skipping")
                    continue
        
        if valid_batches == 0:
            print("Error: No valid batches processed!")
            return None
        
        # Save results
        os.makedirs("./test_results", exist_ok=True)
        with open("./test_results/deeplabv3_plus_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate average metrics from collected scores
        avg_dice = np.mean(test_dice_scores) * 100
        avg_iou = np.mean(test_iou_scores) * 100
        avg_assd = np.mean(test_assd_scores)
        avg_inference_time = np.mean(test_inference_times)
        
        print(f"\nDeepLabV3+ Results (Valid batches: {valid_batches}):")
        print(f"Average Dice (%): {avg_dice:.2f}")
        print(f"Average IoU (%): {avg_iou:.2f}")
        print(f"Average ASSD (1.0 mm): {avg_assd:.2f}")
        print(f"Average Inference Time (s): {avg_inference_time:.4f}")
        
        return {
            'avg_dice': avg_dice,
            'avg_iou': avg_iou,
            'avg_assd': avg_assd,
            'avg_inference_time': avg_inference_time,
            'valid_batches': valid_batches,
            'results': results
        }


if __name__ == "__main__":
    evaluator = DeepLabV3PlusEvaluator()
    results = evaluator.evaluate_dataset()
