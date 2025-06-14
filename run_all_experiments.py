import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        # Remove capture_output=True to show real-time output
        result = subprocess.run(command, shell=True, check=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        return False

def run_experiment(script_name, description):
    """Run a single experiment script"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✓ {description} completed successfully in {duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ Error running {description}: {str(e)}")
        return False

def main():
    """Run complete experimental pipeline"""
    print("Starting comprehensive skin lesion segmentation experiments...")
    
    # Step 1: Data preparation
    if not run_command("python data_preparation.py", "Data preparation"):
        print("Data preparation failed. Exiting...")
        return
    
    # Step 2: Train models
    training_commands = [
        ("python train_yolo.py", "YOLO training"),
        ("python train_unet.py", "U-Net training"),
        ("python train_attention_unet.py", "Attention U-Net training"),
        ("python train_msnet.py", "MSNet training"),
        ("python train_deeplabv3_plus.py", "DeepLabV3+ training")
    ]
    
    for command, description in training_commands:
        if not run_command(command, description):
            print(f"{description} failed. Continuing with existing model if available...")
    
    # Step 3: Test all methods
    test_commands = [
        ("python test_yolo_sam.py", "YOLO+SAM evaluation"),
        ("python test_yolo_sam2.py", "YOLO+SAM2 evaluation"),
        ("python test_yolo_medsam.py", "YOLO+MedSAM evaluation"),
        ("python test_yolo_medsam2.py", "YOLO+MedSAM2 evaluation"),
        ("python test_unet.py", "U-Net evaluation"),
        ("python test_attention_unet.py", "Attention U-Net evaluation"),
        ("python test_msnet.py", "MSNet evaluation"),
        ("python test_deeplabv3_plus.py", "DeepLabV3+ evaluation")
    ]
    
    successful_tests = 0
    for command, description in test_commands:
        if run_command(command, description):
            successful_tests += 1
        else:
            print(f"Warning: {description} failed")
    
    # Step 4: Compare results
    if successful_tests > 0:
        if run_command("python compare_results.py", "Results comparison"):
            print(f"\n{'='*60}")
            print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print(f"Successfully completed {successful_tests}/{len(test_commands)} evaluation methods")
            print("Check the ./test_results/ directory for detailed results and comparisons")
            print('='*60)
        else:
            print("Results comparison failed, but individual results are available")
    else:
        print("No successful evaluations completed")

if __name__ == "__main__":
    main()
