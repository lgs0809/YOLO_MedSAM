import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
from utils.cal_metrics import calculate_dice, calculate_iou, calculate_assd
from deeplabv3_plus.deeplabv3_plus import deeplabv3_plus
from deeplabv3_plus.deeplabv3_training import CE_Loss, Focal_Loss, Dice_loss, weights_init, get_lr_scheduler, set_optimizer_lr

class SkinLesionDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        self.img_dir = os.path.join(data_root, "segmentation_format", split, "images")
        self.mask_dir = os.path.join(data_root, "segmentation_format", split, "masks")
        
        self.images = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((512, 512))(mask)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).long().squeeze(0)  # Binary mask for CE loss
        
        return image, mask

class DeepLabV3PlusTrainer:
    def __init__(self, data_root="./datasets/ISIC2017"):
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = "./saved_models/deeplabv3_plus_skin_lesion.pth"
        os.makedirs("./saved_models", exist_ok=True)
        
        # Loss configuration
        self.cls_weights = None  # Will be calculated from dataset
        self.num_classes = 2  # Background + lesion
    
    
        
    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def train(self, epochs=100, batch_size=8, lr=1e-4, backbone='xception', loss_type='ce'):
        train_transform, val_transform = self.get_transforms()
        
        # Datasets
        train_dataset = SkinLesionDataset(self.data_root, "train", train_transform)
        val_dataset = SkinLesionDataset(self.data_root, "val", val_transform)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Model
        model = deeplabv3_plus(num_classes=self.num_classes, backbone=backbone, pretrained=True)
        model = model.to(self.device)
        
        # Initialize weights - skip for pretrained backbone parts
        # weights_init(model, init_type='kaiming')
        
        # Loss function with better numerical stability
        def stable_criterion(inputs, target):
            if loss_type == 'ce':
                return CE_Loss(inputs, target, self.cls_weights, self.num_classes)
            elif loss_type == 'focal':
                return Focal_Loss(inputs, target, self.cls_weights, self.num_classes)
            elif loss_type == 'dice':
                return Dice_loss(inputs, target)
            else:
                # Use standard CrossEntropyLoss with label smoothing for stability
                inputs = F.interpolate(inputs, size=target.shape[-2:], mode="bilinear", align_corners=True)
                criterion_ce = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)
                return criterion_ce(inputs, target)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        lr_scheduler_func = get_lr_scheduler('cos', lr, lr * 0.01, epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Set learning rate
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            # Training
            model.train()
            train_loss = 0.0
            train_dice_scores = []
            train_iou_scores = []
            train_assd_scores = []
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Ensure outputs are in reasonable range
                outputs = torch.clamp(outputs, -10, 10)
                
                loss = stable_criterion(outputs, masks)
                
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected: {loss.item()}, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate metrics for batch
                with torch.no_grad():
                    pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
                    gt_masks = masks.cpu().numpy()
                    
                    batch_dice = []
                    batch_iou = []
                    batch_assd = []
                    
                    for i in range(pred_masks.shape[0]):
                        pred_mask = pred_masks[i]
                        gt_mask = gt_masks[i]
                        
                        dice = calculate_dice(pred_mask, gt_mask)
                        iou = calculate_iou(pred_mask, gt_mask)
                        assd = calculate_assd(pred_mask, gt_mask)
                        
                        batch_dice.append(dice)
                        batch_iou.append(iou)
                        batch_assd.append(assd)
                    
                    train_dice_scores.extend(batch_dice)
                    train_iou_scores.extend(batch_iou)
                    train_assd_scores.extend(batch_assd)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_dice_scores = []
            val_iou_scores = []
            val_assd_scores = []
            valid_batches = 0
            
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    images, masks = images.to(self.device), masks.to(self.device)
                    
                    outputs = model(images)
                    
                    # Ensure outputs are in reasonable range
                    outputs = torch.clamp(outputs, -10, 10)
                    
                    loss = stable_criterion(outputs, masks)
                    
                    # Check for invalid loss values
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid validation loss detected: {loss.item()}, skipping batch")
                        continue
                    
                    val_loss += loss.item()
                    valid_batches += 1
                    
                    # Calculate metrics for batch
                    pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
                    gt_masks = masks.cpu().numpy()
                    
                    batch_dice = []
                    batch_iou = []
                    batch_assd = []
                    
                    for i in range(pred_masks.shape[0]):
                        pred_mask = pred_masks[i]
                        gt_mask = gt_masks[i]
                        
                        dice = calculate_dice(pred_mask, gt_mask)
                        iou = calculate_iou(pred_mask, gt_mask)
                        assd = calculate_assd(pred_mask, gt_mask)
                        
                        batch_dice.append(dice)
                        batch_iou.append(iou)
                        batch_assd.append(assd)
                    
                    val_dice_scores.extend(batch_dice)
                    val_iou_scores.extend(batch_iou)
                    val_assd_scores.extend(batch_assd)
            
            # Calculate average metrics
            train_loss /= len(train_loader)
            val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')
            
            train_dice_avg = np.mean(train_dice_scores) * 100
            train_iou_avg = np.mean(train_iou_scores) * 100
            train_assd_avg = np.mean(train_assd_scores)
            
            val_dice_avg = np.mean(val_dice_scores) * 100
            val_iou_avg = np.mean(val_iou_scores) * 100
            val_assd_avg = np.mean(val_assd_scores)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice_avg:.2f}%, IoU: {train_iou_avg:.2f}%, ASSD: {train_assd_avg:.2f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice_avg:.2f}%, IoU: {val_iou_avg:.2f}%, ASSD: {val_assd_avg:.2f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.model_save_path)
                print(f"  Best model saved with val loss: {val_loss:.4f}")

if __name__ == "__main__":
    trainer = DeepLabV3PlusTrainer()
    
    print("Starting DeepLabV3+ training...")
    trainer.train(epochs=100, batch_size=4, lr=5e-4, backbone='xception', loss_type='ce')
    
    print("DeepLabV3+ training completed!")
