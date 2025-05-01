
# Road Scene Segmentation for Autonomous Driving

## Project Overview
This project aims to improve road scene segmentation for autonomous vehicles by developing a high-performing deep learning model. Road scene segmentation assigns a semantic label to each pixel in a road scene (e.g., road, vehicle, pedestrian) to help autonomous systems make safe decisions in complex traffic scenarios.

## Objectives
- **Improve Segmentation Accuracy:** Develop a model that achieves high pixel-level classification accuracy (e.g., >80% mIoU) across all relevant classes (roads, vehicles, pedestrians), including small or distant objects.
- **Enhance Boundary Detection:** Produce crisp object boundaries, minimizing ambiguity between adjacent regions (e.g., separating road vs. sidewalk).
- **Optimize Computational Efficiency:** Ensure real-time inference (~10–15 FPS) on standard GPU hardware.

## Methodology
We will implement and compare two state-of-the-art models:
1. **U-Net:** Encoder-decoder architecture with skip connections for improved boundary accuracy and small object segmentation.
2. **DeepLabV3+:** Uses atrous convolutions and an Atrous Spatial Pyramid Pooling (ASPP) module to capture multi-scale context.

### Enhancements
- Introduce an **attention mechanism** to improve small object and boundary segmentation.
- Combine **cross-entropy and Dice loss** to handle class imbalance and improve sharpness.
- Use a **lightweight backbone** (e.g., MobileNet) to reduce computational load and enable real-time processing.

### Training Strategy
- Backpropagation with adaptive learning rates.
- Extensive data augmentation (random scaling, flipping, cropping) to improve generalization.

## Evaluation Metrics
- **Mean Intersection-over-Union (mIoU):** Measures average overlap between predicted and ground truth masks.
- **Dice Coefficient:** Measures the harmonic mean of precision and recall.
- **Per-class IoU:** Assesses how well each object category is segmented.
- **Inference Speed:** Targeting ~10–15 FPS on a standard GPU.

## Datasets
1. **Cityscapes:** 5,000 fine-annotated urban street images from 50 cities (19 classes).
2. **KITTI:** 289 training and 290 test images focused on drivable road area.

<!-- ## Project Structure
```
├── data/                   # Cityscapes and KITTI datasets
├── models/                 # U-Net and DeepLabV3+ models
├── training/               # Training scripts
├── evaluation/             # Evaluation and metrics calculation
├── results/                # Saved models and output visualizations
└── README.md               # Project documentation
``` -->

## Setup
1. Clone the repository:
```bash
git clone https://github.com/poojan-kaneriya/road-scene-segmentation.git
cd road-scene-segmentation
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training
To train the model on Cityscapes:
```bash
python training/train.py --model unet --dataset cityscapes
```

To train the model on KITTI:
```bash
python training/train.py --model deeplab --dataset kitti
```

## Evaluation
To evaluate the model:
```bash
python evaluation/evaluate.py --model unet --dataset cityscapes
```

## Results
- Expected mIoU > 80% on Cityscapes.
- Expected inference speed ~10–15 FPS on GPU.

## Contributors
- Siddharth Mehta
- Poojan Kaneriya

## License
This project is licensed under the MIT License.


## Experiments
| S.N | Architecture | Epochs | Loss Fn | Optimizer | Scheduler | Time per epoch | Training Loss | Validation Loss | Validation mIOU | Additional Comments |
|-----|--------------|--------|---------|-----------|-----------|----------------|---------------|-----------------|-----------------|---------------------|
|1.   |U-Net| 80| Cross Entropy | SGDW (0.01) | Cosine Annealing | 321s | 0.1268 | 0.2086 | 0.4316 | 512x256 sized images after preprocessing with random scaling and cropping added |
|2.   |U-Net| 80| Hybrid Loss | SGDW (0.01) | Cosine Annealing | 340s | 0.1866 |0.2178 | 0.4273| 512x256 sized images after preprocessing with random scaling and cropping added |
|3.   |DeeplabV3+| 100| Cross Entropy | SGDW (0.01) | Cosine Annealing | | | | | Backbone - Resnet50, 512x256 sized images after preprocessing with random scaling and cropping added |
|4.   |DeeplabV3+| 100| Hybrid Loss | SGDW (0.01) | Cosine Annealing |  | | | | Backbone - Resnet50, 512x256 sized images after preprocessing with random scaling and cropping added |