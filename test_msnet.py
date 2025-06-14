import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from train_msnet import SkinLesionDataset
from msnet.miccai_msnet import MSNet
import json
from tqdm import tqdm
import time
from utils.cal_metrics import calculate_dice, calculate_iou, calculate_assd

class MSNetEvaluator:
    def __init__(self, model_path="./saved_models/msnet_skin_lesion.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model
        self.model = MSNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transform for test images
        self.transform = transforms.Compose([
            transforms.Resize((352, 352)),  # MSNet uses 352x352 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict mask for single image"""
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Apply transform (resize to 352x352)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            output = self.model(image_tensor)
            # Apply sigmoid and threshold
            mask_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            mask_binary = (mask_prob > 0.5).astype(np.uint8)
        end_time = time.time()
        
        # Resize mask back to original image size
        mask_resized = cv2.resize(mask_binary, original_size, interpolation=cv2.INTER_NEAREST)
        
        inference_time = end_time - start_time
        return mask_resized, inference_time
    
    def evaluate_dataset(self, test_data_path="./datasets/ISIC2017"):
        """Evaluate MSNet on test dataset"""
        # Create test dataset
        test_dataset = SkinLesionDataset(test_data_path, "test", self.transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        results = []
        
        with torch.no_grad():
            for i, (image, gt_mask) in enumerate(tqdm(test_loader, desc="Evaluating MSNet")):
                # Get original image filename
                img_filename = test_dataset.images[i]
                
                image = image.to(self.device)
                gt_mask = gt_mask.to(self.device)
                
                # Predict with timing
                start_time = time.time()
                output = self.model(image)
                pred_mask = torch.sigmoid(output)
                pred_mask_binary = (pred_mask > 0.5).float()
                end_time = time.time()
                
                inference_time = end_time - start_time
                
                # Convert to numpy
                pred_mask_np = pred_mask_binary.squeeze().cpu().numpy()
                gt_mask_np = gt_mask.squeeze().cpu().numpy()
                
                # Calculate metrics
                dice = calculate_dice(pred_mask_np, gt_mask_np)
                iou = calculate_iou(pred_mask_np, gt_mask_np)
                assd = calculate_assd(pred_mask_np, gt_mask_np)
                
                results.append({
                    'image': img_filename,
                    'dice': float(dice * 100),  # Convert to percentage
                    'iou': float(iou * 100),    # Convert to percentage
                    'assd': float(assd),
                    'inference_time': float(inference_time)
                })
        
        # Save results
        os.makedirs("./test_results", exist_ok=True)
        with open("./test_results/msnet_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate average metrics
        avg_dice = np.mean([r['dice'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        avg_assd = np.mean([r['assd'] for r in results])
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        
        print(f"\nMSNet Results:")
        print(f"Average Dice (%): {avg_dice:.2f}")
        print(f"Average IoU (%): {avg_iou:.2f}")
        print(f"Average ASSD (1.0 mm): {avg_assd:.2f}")
        print(f"Average Inference Time (s): {avg_inference_time:.4f}")
        
        return {
            'avg_dice': avg_dice,
            'avg_iou': avg_iou,
            'avg_assd': avg_assd,
            'avg_inference_time': avg_inference_time,
            'results': results
        }
    

if __name__ == "__main__":
    evaluator = MSNetEvaluator()
    results = evaluator.evaluate_dataset()
