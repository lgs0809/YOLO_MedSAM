import os
import torch
from ultralytics import YOLO
import yaml

class YOLOTrainer:
    def __init__(self, data_path="./datasets/ISIC2017/yolo_format/dataset.yaml"):
        self.data_path = data_path
        self.model_save_path = "./saved_models/yolo_skin_lesion.pt"
        os.makedirs("./saved_models", exist_ok=True)
        
    def train(self, epochs=100, imgsz=640, batch=16, device=0):
        """Train YOLO model"""
        model = YOLO('yolo11l.pt')
        
        # Train the model
        results = model.train(
            data=self.data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project="./runs/detect",
            name="skin_lesion",
            save=True,
            save_period=10,
            val=True,
            patience=20,
            # amp=False,  # Disable AMP to avoid downloading additional models
            # Data augmentation
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            # Optimization
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # resume=True,
        )
        
        # Save the best model
        best_model_path = results.save_dir / "weights" / "best.pt"
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, self.model_save_path)
            print(f"Best model saved to {self.model_save_path}")
        
        return results
    
    def validate(self):
        """Validate trained model"""
        if not os.path.exists(self.model_save_path):
            print("No trained model found. Please train first.")
            return
            
        model = YOLO(self.model_save_path)
        results = model.val(data=self.data_path)
        return results

if __name__ == "__main__":
    trainer = YOLOTrainer()
    
    print("Starting YOLO training...")
    results = trainer.train(epochs=100, batch=16)
    
    print("Validating model...")
    val_results = trainer.validate()
    
    print("YOLO training completed!")
