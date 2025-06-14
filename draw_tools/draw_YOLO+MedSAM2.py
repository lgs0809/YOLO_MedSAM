import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import argparse
from pathlib import Path

class YOLOMedSAM2Visualizer:
    def __init__(self, 
                 yolo_model_path="./saved_models/yolo_skin_lesion.pt",
                 medsam2_model_path="./pretainmodels/MedSAM2_latest.pt",
                 medsam2_config="sam2.1_hiera_t.yaml"):
        """
        Initialize YOLO+MedSAM2 visualizer
        
        Args:
            yolo_model_path: Path to trained YOLO model
            medsam2_model_path: Path to MedSAM2 model
            medsam2_config: MedSAM2 configuration file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load MedSAM2 model
        print("Loading MedSAM2 model...")
        self.sam2_model = build_sam2(medsam2_config, medsam2_model_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        print("Models loaded successfully!")
    
    def detect_with_yolo(self, image, conf_threshold=0.5):
        """
        Detect skin lesions using YOLO
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2] and confidences
        """
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        
        boxes = []
        confidences = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(float(conf))
        
        return boxes, confidences
    
    def segment_with_medsam2(self, image, boxes):
        """
        Segment using MedSAM2 with bounding box prompts
        
        Args:
            image: Input image (numpy array)
            boxes: List of bounding boxes from YOLO
            
        Returns:
            List of segmentation masks
        """
        if not boxes:
            return []
        
        # Set image for SAM2
        self.predictor.set_image(image)
        
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = box
            input_box = np.array([x1, y1, x2, y2])
            
            # Predict mask
            mask, score, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            masks.append(mask[0])
        
        return masks
    
    def visualize_yolo_results(self, image, boxes, confidences):
        """
        Visualize YOLO detection results
        
        Args:
            image: Original image
            boxes: Detected bounding boxes
            confidences: Detection confidences
            
        Returns:
            Image with bounding boxes drawn
        """
        vis_image = image.copy()
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            
            # Draw bounding box with thicker lines
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Add confidence text with larger font
            label = f"Lesion {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 15), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(vis_image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return vis_image
    
    def visualize_medsam2_results(self, image, masks, boxes, confidences):
        """
        Visualize MedSAM2 segmentation results
        
        Args:
            image: Original image
            masks: Segmentation masks
            boxes: Original bounding boxes
            confidences: Detection confidences
            
        Returns:
            Image with segmentation masks overlaid
        """
        vis_image = image.copy()
        
        # Create colored overlay
        overlay = vis_image.copy()
        
        colors = [
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, (mask, box, conf) in enumerate(zip(masks, boxes, confidences)):
            color = colors[i % len(colors)]
            
            # Apply mask overlay
            mask_colored = np.zeros_like(vis_image)
            mask_colored[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, mask_colored, 0.4, 0)
            
            # Draw mask contour with thicker lines
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 4)
        
        return overlay
    
    def visualize_combined_results(self, image, boxes, confidences, masks):
        """
        Visualize combined YOLO detection and MedSAM2 segmentation results
        
        Args:
            image: Original image
            boxes: Detected bounding boxes
            confidences: Detection confidences
            masks: Segmentation masks
            
        Returns:
            Image with both bounding boxes and segmentation masks overlaid
        """
        vis_image = image.copy()
        
        # Create colored overlay for masks
        overlay = vis_image.copy()
        
        colors = [
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, (mask, box, conf) in enumerate(zip(masks, boxes, confidences)):
            color = colors[i % len(colors)]
            
            # Apply mask overlay
            mask_colored = np.zeros_like(vis_image)
            mask_colored[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, mask_colored, 0.3, 0)
            
            # Draw mask contour with thicker lines
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 4)
            
            # Draw bounding box in green with thicker lines
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Add label with larger font
            label = f"Segment {i+1}"
            cv2.putText(overlay, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        return overlay

    def process_image(self, image_path, output_dir="./draw_results", conf_threshold=0.5):
        """
        Process a single image with YOLO+MedSAM2 pipeline
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            conf_threshold: Confidence threshold for YOLO detection
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Step 1: YOLO Detection
        print("Running YOLO detection...")
        boxes, confidences = self.detect_with_yolo(image, conf_threshold)
        print(f"Detected {len(boxes)} lesions")
        
        # Step 2: MedSAM2 Segmentation
        masks = []
        if boxes:
            print("Running MedSAM2 segmentation...")
            masks = self.segment_with_medsam2(image, boxes)
            print(f"Generated {len(masks)} masks")
        else:
            print("No detections found, skipping segmentation")
        
        # Step 3: Visualization
        print("Creating visualizations...")
        
        # YOLO visualization
        yolo_vis = self.visualize_yolo_results(image, boxes, confidences)
        
        # MedSAM2 visualization
        if masks:
            medsam2_vis = self.visualize_medsam2_results(image, masks, boxes, confidences)
            combined_vis = self.visualize_combined_results(image, boxes, confidences, masks)
        else:
            medsam2_vis = image.copy()
            cv2.putText(medsam2_vis, "No detections found", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            combined_vis = medsam2_vis.copy()
        
        # Step 4: Save results
        image_name = Path(image_path).stem
        
        yolo_output_path = os.path.join(output_dir, f"{image_name}_yolo_detection.jpg")
        medsam2_output_path = os.path.join(output_dir, f"{image_name}_medsam2_segmentation.jpg")
        combined_output_path = os.path.join(output_dir, f"{image_name}_combined_results.jpg")
        
        cv2.imwrite(yolo_output_path, yolo_vis)
        cv2.imwrite(medsam2_output_path, medsam2_vis)
        cv2.imwrite(combined_output_path, combined_vis)
        
        print(f"✓ YOLO results saved to: {yolo_output_path}")
        print(f"✓ MedSAM2 results saved to: {medsam2_output_path}")
        print(f"✓ Combined results saved to: {combined_output_path}")
        
        # Step 5: Create summary plot
        self.create_summary_plot(image, yolo_vis, medsam2_vis, combined_vis, output_dir, image_name, 
                                boxes, confidences, masks)
        
        return boxes, confidences, masks
    
    def create_summary_plot(self, original, yolo_vis, medsam2_vis, combined_vis, output_dir, image_name, 
                           boxes, confidences, masks):
        """
        Create a summary plot showing original, YOLO, MedSAM2, and combined results
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Original Image')
        axes[0, 0].axis('off')
        
        # YOLO results
        axes[0, 1].imshow(cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'YOLO Detection\n({len(boxes)} lesions)')
        axes[0, 1].axis('off')
        
        # MedSAM2 results
        axes[1, 0].imshow(cv2.cvtColor(medsam2_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'MedSAM2 Segmentation\n({len(masks)} masks)')
        axes[1, 0].axis('off')
        
        # Combined results
        axes[1, 1].imshow(cv2.cvtColor(combined_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Combined Results\n(Boxes + Masks)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"{image_name}_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Summary plot saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLO+MedSAM2 Image Segmentation Visualization")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./draw_results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for YOLO")
    parser.add_argument("--yolo_model", type=str, default="./saved_models/yolo_skin_lesion.pt", 
                       help="Path to YOLO model")
    parser.add_argument("--medsam2_model", type=str, default="./pretainmodels/MedSAM2_latest.pt", 
                       help="Path to MedSAM2 model")
    parser.add_argument("--medsam2_config", type=str, default="sam2.1_hiera_t.yaml", 
                       help="MedSAM2 config")
    
    args = parser.parse_args()
    
    print("🚀 Starting YOLO+MedSAM2 visualization...")
    print("=" * 60)
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Input image not found: {args.image}")
        return
    
    # Check if models exist
    if not os.path.exists(args.yolo_model):
        print(f"❌ Error: YOLO model not found: {args.yolo_model}")
        print("Please train the YOLO model first using: python train_yolo.py")
        return
    
    if not os.path.exists(args.medsam2_model):
        print(f"❌ Error: MedSAM2 model not found: {args.medsam2_model}")
        print("Please download MedSAM2 model from: https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt")
        return
    
    try:
        # Initialize visualizer
        visualizer = YOLOMedSAM2Visualizer(
            yolo_model_path=args.yolo_model,
            medsam2_model_path=args.medsam2_model,
            medsam2_config=args.medsam2_config
        )
        
        # Process image
        boxes, confidences, masks = visualizer.process_image(
            args.image, 
            args.output, 
            args.conf
        )
        
        print("\n📊 Results Summary:")
        print(f"• Input image: {args.image}")
        print(f"• Detected lesions: {len(boxes)}")
        print(f"• Generated masks: {len(masks)}")
        print(f"• Output directory: {args.output}")
        
        if boxes:
            print("\n🎯 Detection Details:")
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box
                print(f"  Lesion {i+1}: Box=({x1},{y1},{x2},{y2}), Confidence={conf:.3f}")
        
        print("\n✅ Visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage if run without arguments
    import sys
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python draw.py --image path/to/your/image.jpg")
        print("python draw.py --image datasets/ISIC2017/ISIC-2017_Test_v2_Data/ISIC_0000000.jpg --conf 0.3")
    else:
        main()
