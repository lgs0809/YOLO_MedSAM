import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ResultsComparator:
    def __init__(self):
        self.results_dir = "./test_results"
        
    def load_results(self):
        """Load results from all methods"""
        methods = {
            'YOLO+SAM': 'yolo_sam_results.json',
            'YOLO+SAM2': 'yolo_sam2_results.json', 
            'YOLO+MedSAM': 'yolo_medsam_results.json',
            'YOLO+MedSAM2': 'yolo_medsam2_results.json',
            'U-Net': 'unet_results.json',
            'Attention U-Net': 'attention_unet_results.json',
            'MSNet': 'msnet_results.json',
            'DeepLabV3+': 'deeplabv3_plus_results.json'
        }
        
        results = {}
        for method, filename in methods.items():
            filepath = f"{self.results_dir}/{filename}"
            try:
                with open(filepath, 'r') as f:
                    results[method] = json.load(f)
                print(f"Loaded {method} results")
            except FileNotFoundError:
                print(f"Results file not found for {method}: {filepath}")
                results[method] = None
                
        return results
    
    def calculate_summary_metrics(self, results):
        """Calculate summary metrics for each method"""
        summary = {}
        
        for method, data in results.items():
            if data is None:
                continue
                
            # Extract results list
            if 'results' in data:
                all_results = data['results']
            else:
                all_results = data
            
            # Filter out failed detections (dice = 0)
            valid_results = [r for r in all_results if r.get('dice', 0) > 0]
            
            if valid_results:
                avg_dice = np.mean([r['dice'] for r in valid_results])
                avg_iou = np.mean([r['iou'] for r in valid_results])
                avg_assd = np.mean([r['assd'] for r in valid_results])
                avg_inference_time = np.mean([r['inference_time'] for r in valid_results])
                detection_rate = len(valid_results) / len(all_results)
                
                # Convert percentages to ratios if needed (check if values > 1)
                if avg_dice > 1:
                    avg_dice /= 100
                if avg_iou > 1:
                    avg_iou /= 100
            else:
                avg_dice = 0.0
                avg_iou = 0.0
                avg_assd = 100.0
                avg_inference_time = 0.0
                detection_rate = 0.0
                
            summary[method] = {
                'avg_dice': avg_dice,
                'avg_iou': avg_iou,
                'avg_assd': avg_assd,
                'avg_inference_time': avg_inference_time,
                'detection_rate': detection_rate,
                'num_samples': len(all_results)
            }
        
        return summary
    
    def create_comparison_table(self, summary):
        """Create comparison table"""
        df_data = []
        for method, metrics in summary.items():
            df_data.append({
                'Method': method,
                'Dice (%)': f"{metrics['avg_dice']*100:.2f}",
                'IoU (%)': f"{metrics['avg_iou']*100:.2f}",
                'ASSD (1.0 mm)': f"{metrics['avg_assd']:.2f}",
                'Inference Time (s)': f"{metrics['avg_inference_time']:.4f}",
                'Detection Rate': f"{metrics['detection_rate']:.4f}",
                'Samples': metrics['num_samples']
            })
        
        df = pd.DataFrame(df_data)
        return df
    
    def plot_comparison(self, summary):
        """Create comparison plots"""
        methods = list(summary.keys())
        dice_scores = [summary[method]['avg_dice'] for method in methods]
        iou_scores = [summary[method]['avg_iou'] for method in methods]
        assd_scores = [summary[method]['avg_assd'] for method in methods]
        inference_times = [summary[method]['avg_inference_time'] for method in methods]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Color mapping for different method types
        colors = []
        for method in methods:
            if 'DeepLabV3+' in method:
                colors.append('darkred')
            elif 'MSNet' in method:
                colors.append('brown')
            elif 'Attention U-Net' in method:
                colors.append('orange')
            elif 'U-Net' in method:
                colors.append('purple')
            elif 'MedSAM2' in method:
                colors.append('red')
            elif 'MedSAM' in method:
                colors.append('green')
            elif 'SAM2' in method:
                colors.append('cyan')
            elif 'SAM' in method:
                colors.append('blue')
            else:
                colors.append('gray')
        
        # Dice coefficient comparison
        bars1 = axes[0].bar(methods, dice_scores, color=colors)
        axes[0].set_title('Dice Coefficient Comparison')
        axes[0].set_ylabel('Dice Coefficient')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(dice_scores):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # IoU comparison
        bars2 = axes[1].bar(methods, iou_scores, color=colors)
        axes[1].set_title('IoU Comparison')
        axes[1].set_ylabel('IoU')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(iou_scores):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # ASSD comparison (lower is better)
        bars3 = axes[2].bar(methods, assd_scores, color=colors)
        axes[2].set_title('ASSD Comparison (Lower is Better)')
        axes[2].set_ylabel('ASSD (pixels)')
        axes[2].tick_params(axis='x', rotation=45)
        for i, v in enumerate(assd_scores):
            axes[2].text(i, v + max(assd_scores)*0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Inference time comparison (lower is better)
        bars4 = axes[3].bar(methods, inference_times, color=colors)
        axes[3].set_title('Inference Time Comparison (Lower is Better)')
        axes[3].set_ylabel('Inference Time (seconds)')
        axes[3].tick_params(axis='x', rotation=45)
        for i, v in enumerate(inference_times):
            axes[3].text(i, v + max(inference_times)*0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("Loading results...")
        results = self.load_results()
        
        print("Calculating summary metrics...")
        summary = self.calculate_summary_metrics(results)
        
        print("Creating comparison table...")
        comparison_table = self.create_comparison_table(summary)
        
        print("\n" + "="*100)
        print("SKIN LESION SEGMENTATION COMPARISON RESULTS")
        print("="*100)
        print(comparison_table.to_string(index=False))
        print("\n")
        
        # Find best performing methods
        methods_with_results = [m for m in summary.keys() if summary[m]['num_samples'] > 0]
        
        if methods_with_results:
            best_dice_method = max(methods_with_results, key=lambda x: summary[x]['avg_dice'])
            best_iou_method = max(methods_with_results, key=lambda x: summary[x]['avg_iou'])
            best_speed_method = min(methods_with_results, key=lambda x: summary[x]['avg_inference_time'])
            best_assd_method = min(methods_with_results, key=lambda x: summary[x]['avg_assd'])
            
            print("BEST PERFORMERS:")
            print(f"Best Dice Coefficient: {best_dice_method} ({summary[best_dice_method]['avg_dice']*100:.2f}%)")
            print(f"Best IoU: {best_iou_method} ({summary[best_iou_method]['avg_iou']*100:.2f}%)")
            print(f"Best ASSD: {best_assd_method} ({summary[best_assd_method]['avg_assd']:.2f} pixels)")
            print(f"Fastest Inference: {best_speed_method} ({summary[best_speed_method]['avg_inference_time']:.4f}s)")
        
        # Save table
        comparison_table.to_csv(f"{self.results_dir}/comparison_table.csv", index=False)
        
        print("\nGenerating plots...")
        self.plot_comparison(summary)
        
        # Save detailed report
        report = {
            'summary_metrics': summary,
            'comparison_table': comparison_table.to_dict('records')
        }
        
        if methods_with_results:
            report.update({
                'best_dice_method': best_dice_method,
                'best_iou_method': best_iou_method,
                'best_speed_method': best_speed_method,
                'best_assd_method': best_assd_method
            })
        
        with open(f"{self.results_dir}/comparison_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nComparison completed! Results saved in {self.results_dir}/")

if __name__ == "__main__":
    comparator = ResultsComparator()
    comparator.generate_report()
