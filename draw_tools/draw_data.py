import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ISIC2017DataVisualizer:
    def __init__(self, data_root="./datasets/ISIC2017"):
        self.data_root = data_root
        self.splits = ["train", "val", "test"]
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Directory paths
        self.split_dirs = {
            "train": {
                "images": os.path.join(data_root, "ISIC-2017_Training_Data"),
                "masks": os.path.join(data_root, "ISIC-2017_Training_Part1_GroundTruth")
            },
            "val": {
                "images": os.path.join(data_root, "ISIC-2017_Validation_Data"),
                "masks": os.path.join(data_root, "ISIC-2017_Validation_Part1_GroundTruth")
            },
            "test": {
                "images": os.path.join(data_root, "ISIC-2017_Test_v2_Data"),
                "masks": os.path.join(data_root, "ISIC-2017_Test_v2_Part1_GroundTruth")
            }
        }
        
        # Create output directory
        self.output_dir = os.path.join(data_root, "data_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_dataset_statistics(self):
        """Analyze basic dataset statistics"""
        stats = {}
        
        for split in self.splits:
            image_dir = self.split_dirs[split]["images"]
            mask_dir = self.split_dirs[split]["masks"]
            
            if not os.path.exists(image_dir):
                stats[split] = {"images": 0, "masks": 0, "valid_pairs": 0}
                continue
            
            # Count images and masks
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            mask_files = []
            if os.path.exists(mask_dir):
                mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
            
            # Count valid pairs
            valid_pairs = 0
            for img_file in image_files:
                mask_file = img_file.replace('.jpg', '_segmentation.png')
                if mask_file in mask_files:
                    valid_pairs += 1
            
            stats[split] = {
                "images": len(image_files),
                "masks": len(mask_files),
                "valid_pairs": valid_pairs
            }
        
        return stats
    
    def analyze_image_properties(self, max_samples=200):
        """Analyze image properties like size, aspect ratio, etc."""
        properties = {split: [] for split in self.splits}
        
        for split in tqdm(self.splits, desc="Analyzing image properties"):
            image_dir = self.split_dirs[split]["images"]
            mask_dir = self.split_dirs[split]["masks"]
            
            if not os.path.exists(image_dir):
                continue
            
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            # Limit samples for faster processing
            if len(image_files) > max_samples:
                image_files = np.random.choice(image_files, max_samples, replace=False)
            
            for img_file in tqdm(image_files, desc=f"Processing {split}", leave=False):
                img_path = os.path.join(image_dir, img_file)
                mask_file = img_file.replace('.jpg', '_segmentation.png')
                mask_path = os.path.join(mask_dir, mask_file)
                
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    h, w, c = img.shape
                    aspect_ratio = w / h
                    file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
                    
                    # Calculate image statistics
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(img_gray)
                    contrast = np.std(img_gray)
                    
                    # Analyze mask if exists
                    mask_coverage = 0
                    lesion_area = 0
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            mask_binary = (mask > 127).astype(np.uint8)
                            lesion_area = np.sum(mask_binary)
                            mask_coverage = lesion_area / (h * w) * 100
                    
                    properties[split].append({
                        'filename': img_file,
                        'width': w,
                        'height': h,
                        'aspect_ratio': aspect_ratio,
                        'file_size_mb': file_size,
                        'brightness': brightness,
                        'contrast': contrast,
                        'mask_coverage': mask_coverage,
                        'lesion_area': lesion_area
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
        
        return properties
    
    def plot_dataset_overview(self, stats):
        """Plot dataset overview statistics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Data counts
        splits = list(stats.keys())
        image_counts = [stats[split]["images"] for split in splits]
        mask_counts = [stats[split]["masks"] for split in splits]
        valid_pairs = [stats[split]["valid_pairs"] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.25
        
        axes[0].bar(x - width/2, image_counts, width, label='Images', color=self.colors[0])
        axes[0].bar(x + width/2, mask_counts, width, label='Masks', color=self.colors[1])
        # axes[0].bar(x + width, valid_pairs, width, label='Valid Pairs', color=self.colors[2])
        axes[0].set_xlabel('Dataset Split')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Dataset Composition')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(splits)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (img, mask, valid) in enumerate(zip(image_counts, mask_counts, valid_pairs)):
            axes[0].text(i - width, img + max(image_counts) * 0.01, str(img), ha='center', va='bottom')
            axes[0].text(i, mask + max(mask_counts) * 0.01, str(mask), ha='center', va='bottom')
            axes[0].text(i + width, valid + max(valid_pairs) * 0.01, str(valid), ha='center', va='bottom')
        
        # Pie chart for distribution
        total_images = sum(image_counts)
        if total_images > 0:
            axes[1].pie(image_counts, labels=splits, autopct='%1.1f%%', 
                       colors=self.colors[:len(splits)], startangle=90)
            axes[1].set_title('Image Distribution')
        
        # Coverage rate
        coverage_rates = []
        for split in splits:
            if stats[split]["images"] > 0:
                coverage_rates.append(stats[split]["valid_pairs"] / stats[split]["images"] * 100)
            else:
                coverage_rates.append(0)
        
        bars = axes[2].bar(splits, coverage_rates, color=self.colors[3])
        axes[2].set_xlabel('Dataset Split')
        axes[2].set_ylabel('Coverage Rate (%)')
        axes[2].set_title('Mask Coverage Rate')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3)
        
        # Add value labels
        for i, rate in enumerate(coverage_rates):
            axes[2].text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_image_properties(self, properties):
        """Plot image properties analysis"""
        # Combine all data
        all_data = []
        for split, data in properties.items():
            for item in data:
                item['split'] = split
                all_data.append(item)
        
        if not all_data:
            print("No data to plot")
            return
        
        df = pd.DataFrame(all_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Image dimensions distribution
        axes[0].hist([df[df['split'] == split]['width'] for split in self.splits], 
                    bins=30, alpha=0.7, label=self.splits, color=self.colors[:len(self.splits)])
        axes[0].set_xlabel('Image Width (pixels)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Image Width Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Aspect ratio distribution
        axes[1].hist([df[df['split'] == split]['aspect_ratio'] for split in self.splits], 
                    bins=30, alpha=0.7, label=self.splits, color=self.colors[:len(self.splits)])
        axes[1].set_xlabel('Aspect Ratio (W/H)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Aspect Ratio Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. File size distribution
        axes[2].hist([df[df['split'] == split]['file_size_mb'] for split in self.splits], 
                    bins=30, alpha=0.7, label=self.splits, color=self.colors[:len(self.splits)])
        axes[2].set_xlabel('File Size (MB)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('File Size Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Brightness distribution
        axes[3].hist([df[df['split'] == split]['brightness'] for split in self.splits], 
                    bins=30, alpha=0.7, label=self.splits, color=self.colors[:len(self.splits)])
        axes[3].set_xlabel('Average Brightness')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Image Brightness Distribution')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. Contrast distribution
        axes[4].hist([df[df['split'] == split]['contrast'] for split in self.splits], 
                    bins=30, alpha=0.7, label=self.splits, color=self.colors[:len(self.splits)])
        axes[4].set_xlabel('Standard Deviation (Contrast)')
        axes[4].set_ylabel('Frequency')
        axes[4].set_title('Image Contrast Distribution')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # 6. Mask coverage distribution
        mask_data = df[df['mask_coverage'] > 0]
        if not mask_data.empty:
            axes[5].hist([mask_data[mask_data['split'] == split]['mask_coverage'] for split in self.splits], 
                        bins=30, alpha=0.7, label=self.splits, color=self.colors[:len(self.splits)])
            axes[5].set_xlabel('Mask Coverage (%)')
            axes[5].set_ylabel('Frequency')
            axes[5].set_title('Lesion Coverage Distribution')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
        else:
            axes[5].text(0.5, 0.5, 'No mask data available', ha='center', va='center', 
                        transform=axes[5].transAxes, fontsize=14)
            axes[5].set_title('Lesion Coverage Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'image_properties.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_size_analysis(self, properties):
        """Detailed size analysis"""
        # Combine data
        all_data = []
        for split, data in properties.items():
            for item in data:
                item['split'] = split
                all_data.append(item)
        
        if not all_data:
            return
        
        df = pd.DataFrame(all_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Scatter plot: Width vs Height
        for i, split in enumerate(self.splits):
            split_data = df[df['split'] == split]
            if not split_data.empty:
                axes[0, 0].scatter(split_data['width'], split_data['height'], 
                                 alpha=0.6, color=self.colors[i], label=split, s=20)
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Image Dimensions Scatter Plot')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot: Aspect ratios by split
        aspect_data = [df[df['split'] == split]['aspect_ratio'].dropna() for split in self.splits]
        bp1 = axes[0, 1].boxplot(aspect_data, labels=self.splits, patch_artist=True)
        for patch, color in zip(bp1['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 1].set_xlabel('Dataset Split')
        axes[0, 1].set_ylabel('Aspect Ratio')
        axes[0, 1].set_title('Aspect Ratio Distribution by Split')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Common resolutions
        df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
        top_resolutions = df['resolution'].value_counts().head(10)
        
        axes[1, 0].barh(range(len(top_resolutions)), top_resolutions.values, color=self.colors[0])
        axes[1, 0].set_yticks(range(len(top_resolutions)))
        axes[1, 0].set_yticklabels(top_resolutions.index)
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].set_title('Top 10 Image Resolutions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_resolutions.values):
            axes[1, 0].text(v + max(top_resolutions.values) * 0.01, i, str(v), 
                           va='center', ha='left')
        
        # 4. File size vs Image size
        df['total_pixels'] = df['width'] * df['height']
        axes[1, 1].scatter(df['total_pixels'], df['file_size_mb'], alpha=0.6, color=self.colors[1])
        axes[1, 1].set_xlabel('Total Pixels (Width × Height)')
        axes[1, 1].set_ylabel('File Size (MB)')
        axes[1, 1].set_title('File Size vs Image Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'size_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_sample_images(self, num_samples=6):
        """Generate sample images from each split"""
        fig, axes = plt.subplots(len(self.splits), num_samples, figsize=(20, 12))
        if len(self.splits) == 1:
            axes = axes.reshape(1, -1)
        
        for split_idx, split in enumerate(self.splits):
            image_dir = self.split_dirs[split]["images"]
            mask_dir = self.split_dirs[split]["masks"]
            
            if not os.path.exists(image_dir):
                for j in range(num_samples):
                    axes[split_idx, j].text(0.5, 0.5, f'No {split} data', ha='center', va='center',
                                          transform=axes[split_idx, j].transAxes)
                    axes[split_idx, j].set_title(f'{split.upper()} - No Data')
                    axes[split_idx, j].axis('off')
                continue
            
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
            
            for j, img_file in enumerate(sample_files):
                if j >= num_samples:
                    break
                    
                img_path = os.path.join(image_dir, img_file)
                mask_file = img_file.replace('.jpg', '_segmentation.png')
                mask_path = os.path.join(mask_dir, mask_file)
                
                # Load and display image
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Overlay mask if available
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Create colored overlay
                            overlay = img_rgb.copy()
                            mask_colored = np.zeros_like(img_rgb)
                            mask_colored[mask > 127] = [255, 0, 0]  # Red overlay
                            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                            axes[split_idx, j].imshow(overlay)
                            axes[split_idx, j].set_title(f'{split.upper()} - {img_file[:15]}...\nWith Mask')
                        else:
                            axes[split_idx, j].imshow(img_rgb)
                            axes[split_idx, j].set_title(f'{split.upper()} - {img_file[:15]}...\nNo Mask')
                    else:
                        axes[split_idx, j].imshow(img_rgb)
                        axes[split_idx, j].set_title(f'{split.upper()} - {img_file[:15]}...\nNo Mask')
                    
                    axes[split_idx, j].axis('off')
            
            # Fill remaining subplots if not enough samples
            for j in range(len(sample_files), num_samples):
                axes[split_idx, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_statistics_report(self, stats, properties):
        """Generate detailed statistics report"""
        report = {
            "dataset_statistics": stats,
            "image_properties_summary": {}
        }
        
        # Calculate summary statistics for each property
        for split, data in properties.items():
            if not data:
                continue
            
            df = pd.DataFrame(data)
            summary = {
                "count": len(data),
                "dimensions": {
                    "width": {"mean": df['width'].mean(), "std": df['width'].std(), 
                             "min": df['width'].min(), "max": df['width'].max()},
                    "height": {"mean": df['height'].mean(), "std": df['height'].std(),
                              "min": df['height'].min(), "max": df['height'].max()},
                    "aspect_ratio": {"mean": df['aspect_ratio'].mean(), "std": df['aspect_ratio'].std(),
                                   "min": df['aspect_ratio'].min(), "max": df['aspect_ratio'].max()}
                },
                "quality_metrics": {
                    "brightness": {"mean": df['brightness'].mean(), "std": df['brightness'].std()},
                    "contrast": {"mean": df['contrast'].mean(), "std": df['contrast'].std()},
                    "file_size_mb": {"mean": df['file_size_mb'].mean(), "std": df['file_size_mb'].std()}
                }
            }
            
            # Mask coverage statistics
            mask_data = df[df['mask_coverage'] > 0]
            if not mask_data.empty:
                summary["mask_statistics"] = {
                    "coverage_mean": mask_data['mask_coverage'].mean(),
                    "coverage_std": mask_data['mask_coverage'].std(),
                    "coverage_min": mask_data['mask_coverage'].min(),
                    "coverage_max": mask_data['mask_coverage'].max(),
                    "samples_with_mask": len(mask_data)
                }
            
            report["image_properties_summary"][split] = summary
        
        # Save report
        with open(os.path.join(self.output_dir, 'statistics_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def run_complete_analysis(self):
        """Run complete data analysis and visualization"""
        print("🚀 Starting ISIC2017 Data Analysis...")
        print("=" * 60)
        
        # 1. Basic statistics
        print("📊 Step 1: Analyzing dataset statistics...")
        stats = self.analyze_dataset_statistics()
        
        # 2. Image properties
        print("🔍 Step 2: Analyzing image properties...")
        properties = self.analyze_image_properties()
        
        # 3. Generate visualizations
        print("📈 Step 3: Generating visualizations...")
        
        print("  - Dataset overview...")
        self.plot_dataset_overview(stats)
        
        print("  - Image properties...")
        self.plot_image_properties(properties)
        
        print("  - Size analysis...")
        self.plot_size_analysis(properties)
        
        print("  - Sample images...")
        self.generate_sample_images()
        
        # 4. Generate report
        print("📝 Step 4: Generating statistics report...")
        report = self.generate_statistics_report(stats, properties)
        
        # 5. Print summary
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_images = sum([stats[split]["images"] for split in self.splits])
        total_masks = sum([stats[split]["masks"] for split in self.splits])
        total_valid_pairs = sum([stats[split]["valid_pairs"] for split in self.splits])
        
        print(f"Total Images: {total_images}")
        print(f"Total Masks: {total_masks}")
        print(f"Valid Pairs: {total_valid_pairs}")
        print(f"Overall Coverage: {total_valid_pairs/total_images*100:.1f}%" if total_images > 0 else "N/A")
        
        for split in self.splits:
            print(f"\n{split.upper()} Split:")
            print(f"  Images: {stats[split]['images']}")
            print(f"  Valid Pairs: {stats[split]['valid_pairs']}")
            if split in report["image_properties_summary"] and report["image_properties_summary"][split]:
                prop_summary = report["image_properties_summary"][split]
                print(f"  Avg Dimensions: {prop_summary['dimensions']['width']['mean']:.0f}x{prop_summary['dimensions']['height']['mean']:.0f}")
                print(f"  Avg File Size: {prop_summary['quality_metrics']['file_size_mb']['mean']:.2f} MB")
                if "mask_statistics" in prop_summary:
                    print(f"  Avg Lesion Coverage: {prop_summary['mask_statistics']['coverage_mean']:.1f}%")
        
        print(f"\n✅ Analysis completed! Results saved in: {self.output_dir}")
        print("=" * 60)

if __name__ == "__main__":
    # Initialize visualizer
    visualizer = ISIC2017DataVisualizer()
    
    # Run complete analysis
    visualizer.run_complete_analysis()
