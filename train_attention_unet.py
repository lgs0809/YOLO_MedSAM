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
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()  # Binary mask
        
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        d5 = self.up1(x5)
        x4 = self.att1(g=d5, x=x4)
        d5 = torch.cat([x4, d5], dim=1)
        d5 = self.conv_up1(d5)
        
        d4 = self.up2(d5)
        x3 = self.att2(g=d4, x=x3)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.conv_up2(d4)
        
        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv_up3(d3)
        
        d2 = self.up4(d3)
        x1 = self.att4(g=d2, x=x1)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.conv_up4(d2)
        
        logits = self.outc(d2)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # weight for BCE
        self.beta = beta    # weight for Dice
        self.gamma = gamma  # weight for Focal
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        return self.alpha * bce + self.beta * dice + self.gamma * focal

class AttentionUNetTrainer:
    def __init__(self, data_root="./datasets/ISIC2017"):
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = "./saved_models/attention_unet_skin_lesion.pth"
        os.makedirs("./saved_models", exist_ok=True)
        
    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def train(self, epochs=100, batch_size=8, lr=1e-4):
        train_transform, val_transform = self.get_transforms()
        
        # Datasets
        train_dataset = SkinLesionDataset(self.data_root, "train", train_transform)
        val_dataset = SkinLesionDataset(self.data_root, "val", val_transform)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Model
        model = AttentionUNet(n_channels=3, n_classes=1).to(self.device)
        
        # Loss and optimizer
        criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        best_val_loss = float('inf')
        
        print(f"Training Attention U-Net on {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(epochs):
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
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate metrics for batch
                with torch.no_grad():
                    pred_masks = torch.sigmoid(outputs)
                    pred_masks_binary = (pred_masks > 0.5).cpu().numpy()
                    gt_masks = masks.cpu().numpy()
                    
                    batch_dice = []
                    batch_iou = []
                    batch_assd = []
                    
                    for i in range(pred_masks_binary.shape[0]):
                        pred_mask = pred_masks_binary[i].squeeze()
                        gt_mask = gt_masks[i].squeeze()
                        
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
            
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    images, masks = images.to(self.device), masks.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    
                    # Calculate metrics for batch
                    pred_masks = torch.sigmoid(outputs)
                    pred_masks_binary = (pred_masks > 0.5).cpu().numpy()
                    gt_masks = masks.cpu().numpy()
                    
                    batch_dice = []
                    batch_iou = []
                    batch_assd = []
                    
                    for i in range(pred_masks_binary.shape[0]):
                        pred_mask = pred_masks_binary[i].squeeze()
                        gt_mask = gt_masks[i].squeeze()
                        
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
            val_loss /= len(val_loader)
            
            train_dice_avg = np.mean(train_dice_scores) * 100  # Convert to percentage
            train_iou_avg = np.mean(train_iou_scores) * 100    # Convert to percentage
            train_assd_avg = np.mean(train_assd_scores)
            
            val_dice_avg = np.mean(val_dice_scores) * 100      # Convert to percentage
            val_iou_avg = np.mean(val_iou_scores) * 100        # Convert to percentage
            val_assd_avg = np.mean(val_assd_scores)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice_avg:.2f}%, IoU: {train_iou_avg:.2f}%, ASSD: {train_assd_avg:.2f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice_avg:.2f}%, IoU: {val_iou_avg:.2f}%, ASSD: {val_assd_avg:.2f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.model_save_path)
                print(f"  Best model saved with val loss: {val_loss:.4f}")

if __name__ == "__main__":
    trainer = AttentionUNetTrainer()
    
    print("Starting Attention U-Net training...")
    trainer.train(epochs=100, batch_size=4, lr=1e-4)
    
    print("Attention U-Net training completed!")
