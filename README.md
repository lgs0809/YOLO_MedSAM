# YOLO-SAM: Comprehensive Skin Lesion Segmentation Framework

A comprehensive framework comparing different approaches for skin lesion segmentation, combining YOLO object detection with various SAM models and traditional segmentation methods.

## 🚀 Overview

This project implements and compares multiple approaches for medical image segmentation:

- **YOLO + SAM**: Object detection followed by segmentation
- **YOLO + SAM2**: Enhanced version with SAM2
- **YOLO + MedSAM**: Medical-specific SAM variant
- **YOLO + MedSAM2**: Medical-specific SAM2 variant
- **U-Net**: Traditional CNN-based segmentation
- **Attention U-Net**: U-Net with attention mechanisms
- **MSNet**: Multi-scale Subtraction Network for polyp segmentation
- **DeepLabV3+**: Segmentation with strous separable convolution

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🔧 Installation

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/lgs0809/YOLO_MedSAM.git
cd YOLO_MedSAM
```

2. **Download pre-trained models**

    Download SAMs models and place them in the `pretrained_models/` directory:
    - **SAM (vit-b)**: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    - **SAM2 (tiny)**: [SAM2 model (tiny)](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
    - **MedSAM**: [MedSAM model](https://github.com/bowang-lab/MedSAM?tab=readme-ov-file)
    - **MedSAM2**: [MedSAM2 model](https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt)

    YOLO models will be downloaded during first training

## 📊 Dataset

The project uses the **ISIC 2017** skin lesion dataset:

- **Training**: 2,000 dermoscopic images
- **Validation**: 150 images  
- **Test**: 600 images
- **Task**: Binary segmentation (lesion vs background)

### Data Preparation

Run the automated data preparation script:

```bash
python data_preparation.py
```

This will:
- Download ISIC 2017 dataset
- Convert to YOLO format (bounding boxes)
- Prepare segmentation format (512×512 images)

## 🚀 Quick Start

### Run All Experiments

Execute the complete experimental pipeline:

```bash
python run_all_experiments.py
```

This will:
1. Prepare data
2. Train all models
3. Evaluate all methods
4. Generate comparison results

### Individual Model Training

```bash
# Train YOLO detector
python train_yolo.py

# Train U-Net
python train_unet.py

# Train Attention U-Net
python train_attention_unet.py

# Train MSNet
python train_msnet.py

# Train DeepLabV3+
python train_deeplabv3_plus.py
```

### Individual Model Testing

```bash
# Test YOLO + SAM
python test_yolo_sam.py

# Test YOLO + SAM2
python test_yolo_sam2.py

# Test YOLO + MedSAM
python test_yolo_medsam.py

# Test YOLO + MedSAM2
python test_yolo_medsam2.py

# Test U-Net
python test_unet.py

# Test Attention U-Net
python test_attention_unet.py

# Test MSNet
python test_msnet.py

# Test DeepLabV3+
python test_deeplabv3_plus.py
```

### Test All Models
```bash
python test_all.py
```

### Results Comparison
```bash
python compare_results.py
```

## 🏗️ Model Architectures

### 1. YOLO + SAM Pipeline

```
Input Image → YOLO Detection → Bounding Box → SAM Segmentation → Final Mask
```

- **YOLO**: Object detection for lesion localization
- **SAM**: Segment Anything Model for precise segmentation
- **Variants**: SAM, SAM2, MedSAM, MedSAM2

### 2. Traditional Segmentation

- **U-Net**: Encoder-decoder with skip connections
- **Attention U-Net**: U-Net enhanced with attention gates
- **MSNet**: Multi-scale Subtraction Network
- **DeepLabV3+**: Encoder-Decoder with atrous separable convolution 

## 📈 Evaluation

### Metrics

All models are evaluated using:

- **Dice Coefficient**: Overlap similarity
- **IoU (Jaccard Index)**: Intersection over Union
- **ASSD (Average Symmetric Surface Distance)**: Distance between predicted and ground truth boundaries
- **Inference Time**: Speed of segmentation

Results are saved in `./test_results/` directory with:
- Individual model metrics
- Comparison tables
- Visualization plots

## 📊 Results

### Performance Comparison

| Method | Dice (%)↑ | IoU (%)↑ | ASSD (1.0 mm)↓ | Inference Time (s)↓ |
|--------|------|-----|----------|-------------|
| YOLO+SAM (vit-b) | 81.92 | 71.13 | 3.25 | 0.1947 |
| YOLO+SAM2 (tiny) | 81.15 | 71.11 | 3.40 | 0.0994 |
| YOLO+MedSAM | 81.21 | 70.81 | 3.12 | 0.1949 |
| YOLO+MedSAM2 | 88.06 | 80.57 | 2.22 | 0.1000 |
| U-Net | 74.17 | 63.86 | 6.86 | 0.0027 |
| Attention U-Net | 76.67 | 65.88 | 5.91 | 0.0038 |
| MSNet | 79.55 | 68.98 | 3.44 | 0.0145|

*Note: Results may vary based on random initialization and data process*

## 📁 Project Structure

```
yolo_sam/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── run_all_experiments.py        # Complete pipeline runner
├── data_preparation.py           # Dataset download and preparation
├── compare_results.py            # Results comparison and analysis
├── test_all.py                   # Test all models at once
│
├── Training Scripts/
├── train_yolo.py                # YOLO training
├── train_unet.py                # U-Net training
├── train_attention_unet.py      # Attention U-Net training
├── train_msnet.py               # MSNet training
├── train_deeplabv3_plus.py      # DeepLabV3+ training
│
├── Testing Scripts/
├── test_yolo_sam.py             # YOLO+SAM evaluation
├── test_yolo_sam2.py            # YOLO+SAM2 evaluation
├── test_yolo_medsam.py          # YOLO+MedSAM evaluation
├── test_yolo_medsam2.py         # YOLO+MedSAM2 evaluation
├── test_unet.py                 # U-Net evaluation
├── test_attention_unet.py       # Attention U-Net evaluation
├── test_msnet.py                # MSNet evaluation
├── test_deeplabv3_plus.py       # DeepLabV3+ evaluation
│
├── Models/
├── saved_models/                # Saved model checkpoints
├── segment_anything/            # SAM implementation
├── sam2/                        # SAM2 implementation
├── pretrained_models/           # Pre-trained model weights
│   ├── sam_vit_b_01ec64.pth    # SAM ViT-B weights
│   ├── sam2.1_hiera_tiny.pt    # SAM2 tiny weights
│   ├── MedSAM_latest.pt        # MedSAM weights
│   └── MedSAM2_latest.pt       # MedSAM2 weights
│
├── Data/
├── datasets/                    # Dataset storage
│   └── ISIC2017/               # ISIC 2017 dataset
│       ├── yolo_format/        # YOLO format data
│       │   ├── train/          # Training images and labels
│       │   ├── valid/          # Validation images and labels
│       │   └── test/           # Test images and labels
│       └── segmentation_format/ # Segmentation format data
│           ├── train/          # Training images and masks
│           ├── valid/          # Validation images and masks
│           └── test/           # Test images and masks
│
└── Results/
    └── test_results/           # Evaluation results
        ├── yolo_sam_results.json       # YOLO+SAM results
        ├── yolo_sam2_results.json      # YOLO+SAM2 results
        ├── yolo_medsam_results.json    # YOLO+MedSAM results
        ├── yolo_medsam2_results.json   # YOLO+MedSAM2 results
        ├── unet_results.json           # U-Net results
        ├── attention_unet_results.json # Attention U-Net results
        ├── msnet_results.json          # MSNet results
        ├── deeplabv3_plus_results.json # DeepLabV3+ results
        ├── comparison_table.csv        # Performance comparison
        └── comparison_plots.png        # Visualization plots
```

## 🎛️ Configuration

### Adding New Models

1. Create training script: `train_your_model.py`
2. Create testing script: `test_your_model.py`
3. Add to `run_all_experiments.py`
4. Update `test_all.py` and `compare_results.py`

## 🔬 Technical Details

### SAM Integration

- **Prompt Engineering**: Bounding box prompts from YOLO
- **Post-processing**: Confidence filtering and mask refinement
- **Model Variants**: Support for SAM, SAM2, MedSAM, MedSAM2

## 📚 References

1. **SAM**: [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)
2. **SAM2**: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
3. **MedSAM&MedSAM2**: [Medical SAM2](https://doi.org/10.1038/s41467-024-44824-z)
4. **YOLOv11**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
5. **U-Net**: [Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28)
6. **MSNet**: [Automatic Polyp Segmentation via Multi-scale Subtraction Network](https://doi.org/10.1007/978-3-030-87193-2_12)
7. **DeepLabV3+**: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://doi.org/10.48550/arXiv.1802.02611)
8. **ISIC 2017**: [ISIC 2017 Challenge](https://challenge.isic-archive.com/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ISIC 2017 Challenge organizers for the dataset
- Meta AI for SAM and SAM2 models
- bowang-lab for MedSAM and MedSAM2 models
- Ultralytics for YOLO implementation
- PyTorch team for the deep learning framework


**🎉 Happy Segmenting! 🎉**