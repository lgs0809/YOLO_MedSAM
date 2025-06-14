import os
import sys
import time
import json
import traceback
from datetime import datetime

def run_test(test_module, test_name):
    """Run a single test and return results"""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Module: {test_module}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        # Import and run test
        if test_module == "test_yolo_sam":
            from test_yolo_sam import YOLOSAMPredictor
            predictor = YOLOSAMPredictor()
            results = predictor.evaluate_dataset()
            
        elif test_module == "test_yolo_sam2":
            from test_yolo_sam2 import YOLOSAM2Predictor
            predictor = YOLOSAM2Predictor()
            results = predictor.evaluate_dataset()
            
        elif test_module == "test_yolo_medsam":
            from test_yolo_medsam import YOLOMedSAMPredictor
            predictor = YOLOMedSAMPredictor()
            results = predictor.evaluate_dataset()
            
        elif test_module == "test_yolo_medsam2":
            from test_yolo_medsam2 import YOLOSAM2Predictor
            predictor = YOLOSAM2Predictor()
            results = predictor.evaluate_dataset()
            
        elif test_module == "test_unet":
            from test_unet import UNetEvaluator
            evaluator = UNetEvaluator()
            results = evaluator.evaluate_dataset()
            
        elif test_module == "test_attention_unet":
            from test_attention_unet import AttentionUNetEvaluator
            evaluator = AttentionUNetEvaluator()
            results = evaluator.evaluate_dataset()
            
        elif test_module == "test_msnet":
            from test_msnet import MSNetEvaluator
            evaluator = MSNetEvaluator()
            results = evaluator.evaluate_dataset()
            
        elif test_module == "test_deeplabv3_plus":
            from test_deeplabv3_plus import DeepLabV3PlusEvaluator
            evaluator = DeepLabV3PlusEvaluator()
            results = evaluator.evaluate_dataset()
            
        else:
            print(f"Unknown test module: {test_module}")
            return None
        
        end_time = time.time()
        duration = end_time - start_time
        
        if results:
            print(f"\n✓ {test_name} completed successfully in {duration:.2f} seconds")
            return {
                'test_name': test_name,
                'module': test_module,
                'duration': duration,
                'results': results,
                'status': 'success'
            }
        else:
            print(f"\n⚠ {test_name} completed but returned no results")
            return {
                'test_name': test_name,
                'module': test_module,
                'duration': duration,
                'results': None,
                'status': 'no_results'
            }
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✗ {test_name} failed with error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        
        return {
            'test_name': test_name,
            'module': test_module,
            'duration': duration,
            'error': str(e),
            'status': 'failed'
        }

def main():
    """Run all tests and compile results"""
    print("Starting comprehensive evaluation of all segmentation methods...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define all tests
    all_tests = [
        ("test_yolo_sam", "YOLO+SAM Evaluation"),
        ("test_yolo_sam2", "YOLO+SAM2 Evaluation"),
        ("test_yolo_medsam", "YOLO+MedSAM Evaluation"),
        ("test_yolo_medsam2", "YOLO+MedSAM2 Evaluation"),
        ("test_unet", "U-Net Evaluation"),
        ("test_attention_unet", "Attention U-Net Evaluation"),
        ("test_msnet", "MSNet Evaluation"),
        ("test_deeplabv3_plus", "DeepLabV3+ Evaluation")
    ]
    
    all_results = []
    successful_tests = 0
    total_start_time = time.time()
    
    # Run each test
    for test_module, test_name in all_tests:
        result = run_test(test_module, test_name)
        all_results.append(result)
        
        if result and result['status'] == 'success':
            successful_tests += 1
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Compile summary results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print('='*80)
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Successful tests: {successful_tests}/{len(all_tests)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display results for successful tests
    print(f"\n{'Method':<20} {'Dice (%)':<10} {'IoU (%)':<10} {'ASSD (1.0 mm)':<12} {'Time (s)':<10} {'Status'}")
    print('-' * 85)
    
    summary_data = {}
    
    for result in all_results:
        if result['status'] == 'success' and result['results']:
            method_name = result['test_name'].replace(' Evaluation', '')
            res = result['results']
            
            # Extract metrics (all methods should have these four metrics)
            dice = res.get('avg_dice', 0)
            iou = res.get('avg_iou', 0)
            assd = res.get('avg_assd', 0)
            inference_time = res.get('avg_inference_time', 0)
            
            print(f"{method_name:<20} {dice:<10.2f} {iou:<10.2f} {assd:<12.2f} {inference_time:<10.4f} Success")
            
            summary_data[method_name] = {
                'dice': dice,
                'iou': iou,
                'assd': assd,
                'inference_time': inference_time
            }
                
        elif result['status'] == 'failed':
            method_name = result['test_name'].replace(' Evaluation', '')
            print(f"{method_name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} Failed")
            
        elif result['status'] == 'no_results':
            method_name = result['test_name'].replace(' Evaluation', '')
            print(f"{method_name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} No Results")
    
    # Find best performing methods
    if summary_data:
        print(f"\n{'='*60}")
        print("BEST PERFORMING METHODS")
        print('='*60)
        
        # Best Dice score
        best_dice_method = max(summary_data.items(), key=lambda x: x[1]['dice'])
        print(f"Best Dice Score: {best_dice_method[0]} ({best_dice_method[1]['dice']:.2f}%)")
        
        # Best IoU score
        best_iou_method = max(summary_data.items(), key=lambda x: x[1]['iou'])
        print(f"Best IoU Score: {best_iou_method[0]} ({best_iou_method[1]['iou']:.2f}%)")
        
        # Best ASSD (lowest is better)
        best_assd_method = min(summary_data.items(), key=lambda x: x[1]['assd'])
        print(f"Best ASSD: {best_assd_method[0]} ({best_assd_method[1]['assd']:.2f} 1.0 mm)")
        
        # Fastest inference
        fastest_method = min(summary_data.items(), key=lambda x: x[1]['inference_time'])
        print(f"Fastest Inference: {fastest_method[0]} ({fastest_method[1]['inference_time']:.4f}s)")
    
    # Save comprehensive results
    os.makedirs("./test_results", exist_ok=True)
    
    comprehensive_results = {
        'timestamp': datetime.now().isoformat(),
        'total_duration': total_duration,
        'successful_tests': successful_tests,
        'total_tests': len(all_tests),
        'summary_data': summary_data,
        'detailed_results': all_results
    }
    
    with open("./test_results/comprehensive_evaluation.json", 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\nComprehensive results saved to ./test_results/comprehensive_evaluation.json")
    
    if successful_tests == len(all_tests):
        print(f"\n🎉 All {len(all_tests)} tests completed successfully!")
    elif successful_tests > 0:
        print(f"\n⚠ {successful_tests}/{len(all_tests)} tests completed successfully")
    else:
        print(f"\n❌ No tests completed successfully")
    
    return comprehensive_results

if __name__ == "__main__":
    results = main()
