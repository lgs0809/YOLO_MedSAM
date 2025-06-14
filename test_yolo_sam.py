import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import json
import time
from utils.cal_metrics import calculate_dice, calculate_iou, calculate_assd

class YOLOSAMPredictor:
    def __init__(self, 
                 yolo_model_path="./saved_models/yolo_skin_lesion.pt",
                 sam_checkpoint_path="./pretainmodels/sam_vit_b_01ec64.pth",
                 sam_model_type="vit_b"):
        
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load SAM model
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = SamPredictor(self.sam)
        
    def predict(self, image_path, conf_threshold=0.5):
        """Predict using YOLO+SAM pipeline"""
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        
        # YOLO detection
        yolo_results = self.yolo_model(image_path, conf=conf_threshold, verbose=False)
        
        if len(yolo_results[0].boxes) == 0:
            end_time = time.time()
            return None, None, None, end_time - start_time
        
        # Set image for SAM
        self.predictor.set_image(image_rgb)
        
        # Get best detection box
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        confidences = yolo_results[0].boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        best_box = boxes[best_idx]
        
        # SAM prediction with box prompt
        masks, scores, logits = self.predictor.predict(
            box=best_box,
            multimask_output=True
        )
        
        # Select best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        return best_mask, best_box, scores[best_mask_idx], inference_time
    
    def evaluate_dataset(self, test_data_path="./datasets/ISIC2017/segmentation_format/test"):
        """Evaluate on test dataset"""
        test_img_dir = os.path.join(test_data_path, "images")
        test_mask_dir = os.path.join(test_data_path, "masks")
        
        if not os.path.exists(test_img_dir):
            print(f"Test directory not found: {test_img_dir}")
            return
        
        results = []
        image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
        
        for img_file in tqdm(image_files, desc="Evaluating YOLO+SAM"):
            img_path = os.path.join(test_img_dir, img_file)
            mask_file = img_file.replace('.jpg', '_segmentation.png')
            gt_mask_path = os.path.join(test_mask_dir, mask_file)
            
            if not os.path.exists(gt_mask_path):
                continue
            
            # Predict
            pred_mask, bbox, confidence, inference_time = self.predict(img_path)
            
            if pred_mask is None:
                results.append({
                    'image': img_file,
                    'dice': 0.0,
                    'iou': 0.0,
                    'assd': 100.0,
                    'inference_time': inference_time
                })
                continue
            
            # Load ground truth
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, pred_mask.shape[::-1])
            gt_mask = (gt_mask > 127).astype(np.uint8)
            
            # Calculate metrics
            dice = calculate_dice(pred_mask.astype(np.uint8), gt_mask)
            iou = calculate_iou(pred_mask.astype(np.uint8), gt_mask)
            assd = calculate_assd(pred_mask.astype(np.uint8), gt_mask)
            
            results.append({
                'image': img_file,
                'dice': float(dice * 100),  # Convert to percentage
                'iou': float(iou * 100),    # Convert to percentage
                'assd': float(assd),
                'inference_time': float(inference_time)
            })
        
        # Save results
        os.makedirs("./test_results", exist_ok=True)
        with open("./test_results/yolo_sam_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate average metrics
        avg_dice = np.mean([r['dice'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        avg_assd = np.mean([r['assd'] for r in results])
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        
        print(f"\nYOLO+SAM Results:")
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
    predictor = YOLOSAMPredictor()
    results = predictor.evaluate_dataset()
