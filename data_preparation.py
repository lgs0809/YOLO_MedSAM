import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import requests
import zipfile
from tqdm import tqdm
import shutil

class ISIC2017DataProcessor:
    def __init__(self, data_root="./datasets/ISIC2017"):
        self.data_root = data_root
        self.train_dir = os.path.join(data_root, "ISIC-2017_Training_Data")
        self.train_mask_dir = os.path.join(data_root, "ISIC-2017_Training_Part1_GroundTruth")
        self.val_dir = os.path.join(data_root, "ISIC-2017_Validation_Data") 
        self.val_mask_dir = os.path.join(data_root, "ISIC-2017_Validation_Part1_GroundTruth")
        self.test_dir = os.path.join(data_root, "ISIC-2017_Test_v2_Data")
        self.test_mask_dir = os.path.join(data_root, "ISIC-2017_Test_v2_Part1_GroundTruth")
        
        os.makedirs(data_root, exist_ok=True)
        
    def download_dataset(self):
        """Download ISIC2017 dataset"""
        urls = {
            "train_images": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip",
            "train_masks": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip",
            "val_images": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip",
            "val_masks": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip",
            "test_images": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip",
            "test_masks": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
        }
        
        for name, url in tqdm(urls.items(), desc="Overall Download Progress"):
            zip_path = os.path.join(self.data_root, f"{name}.zip")
            if not os.path.exists(zip_path):
                print(f"\nDownloading {name}...")
                
                # Get file size for progress bar
                response = requests.head(url)
                total_size = int(response.headers.get('content-length', 0))
                
                response = requests.get(url, stream=True)
                
                with open(zip_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"Downloading {name}", leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                print(f"Extracting {name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    files = zip_ref.namelist()
                    with tqdm(total=len(files), desc=f"Extracting {name}", leave=False) as pbar:
                        for file in files:
                            zip_ref.extract(file, self.data_root)
                            pbar.update(1)
                
                os.remove(zip_path)
                print(f"✓ {name} completed")
            else:
                print(f"✓ {name} already exists, skipping...")

    def mask_to_bbox(self, mask_path):
        """Convert binary mask to bounding box"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Get bounding box from largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return [x, y, x + w, y + h]  # [x1, y1, x2, y2]
    
    def prepare_yolo_data(self):
        """Prepare data in YOLO format"""
        yolo_root = os.path.join(self.data_root, "yolo_format")
        os.makedirs(yolo_root, exist_ok=True)
        
        splits = ["train", "val", "test"]
        
        for split in tqdm(splits, desc="Processing YOLO splits"):
            img_dir = getattr(self, f"{split}_dir")
            mask_dir = getattr(self, f"{split}_mask_dir")
            
            # Create YOLO directories
            yolo_img_dir = os.path.join(yolo_root, "images", split)
            yolo_label_dir = os.path.join(yolo_root, "labels", split)
            os.makedirs(yolo_img_dir, exist_ok=True)
            os.makedirs(yolo_label_dir, exist_ok=True)
            
            if not os.path.exists(img_dir):
                continue
                
            image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
            
            processed_count = 0
            for img_file in tqdm(image_files, desc=f"Processing {split} for YOLO", leave=False):
                img_path = os.path.join(img_dir, img_file)
                mask_file = img_file.replace('.jpg', '_segmentation.png')
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    continue
                
                # Copy image
                dst_img_path = os.path.join(yolo_img_dir, img_file)
                shutil.copy2(img_path, dst_img_path)
                
                # Generate YOLO label
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                
                bbox = self.mask_to_bbox(mask_path)
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = bbox
                # Convert to YOLO format (normalized center coordinates and dimensions)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Save YOLO label (class 0 for skin lesion)
                label_file = img_file.replace('.jpg', '.txt')
                label_path = os.path.join(yolo_label_dir, label_file)
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                processed_count += 1
            
            print(f"✓ {split}: {processed_count} images processed")
        
        # Create dataset.yaml for YOLO
        yaml_content = f"""
path: {yolo_root}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['skin_lesion']
"""
        with open(os.path.join(yolo_root, "dataset.yaml"), 'w') as f:
            f.write(yaml_content)
    
    def prepare_segmentation_data(self):
        """Prepare data for segmentation models"""
        seg_root = os.path.join(self.data_root, "segmentation_format")
        os.makedirs(seg_root, exist_ok=True)
        
        splits = ["train", "val", "test"]
        
        for split in tqdm(splits, desc="Processing segmentation splits"):
            img_dir = getattr(self, f"{split}_dir")
            mask_dir = getattr(self, f"{split}_mask_dir")
            
            # Create segmentation directories
            seg_img_dir = os.path.join(seg_root, split, "images")
            seg_mask_dir = os.path.join(seg_root, split, "masks")
            os.makedirs(seg_img_dir, exist_ok=True)
            os.makedirs(seg_mask_dir, exist_ok=True)
            
            if not os.path.exists(img_dir):
                continue
                
            image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
            
            processed_count = 0
            for img_file in tqdm(image_files, desc=f"Processing {split} for segmentation", leave=False):
                img_path = os.path.join(img_dir, img_file)
                mask_file = img_file.replace('.jpg', '_segmentation.png')
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    continue
                
                # Copy and resize image to 512x512
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (512, 512))
                dst_img_path = os.path.join(seg_img_dir, img_file)
                cv2.imwrite(dst_img_path, img_resized)
                
                # Copy and resize mask to 512x512
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_resized = cv2.resize(mask, (512, 512))
                # Ensure binary mask (0 or 255)
                mask_resized = (mask_resized > 127).astype(np.uint8) * 255
                dst_mask_path = os.path.join(seg_mask_dir, mask_file)
                cv2.imwrite(dst_mask_path, mask_resized)
                
                processed_count += 1
            
            print(f"✓ {split}: {processed_count} images processed")

if __name__ == "__main__":
    processor = ISIC2017DataProcessor()
    
    print("🚀 Starting ISIC2017 dataset preparation...")
    print("=" * 50)
    
    print("\n📥 Step 1: Downloading ISIC2017 dataset...")
    processor.download_dataset()
    
    print("\n🎯 Step 2: Preparing YOLO format data...")
    processor.prepare_yolo_data()
    
    print("\n🎨 Step 3: Preparing segmentation format data...")
    processor.prepare_segmentation_data()
    
    print("\n✅ Data preparation completed successfully!")
    print("=" * 50)
