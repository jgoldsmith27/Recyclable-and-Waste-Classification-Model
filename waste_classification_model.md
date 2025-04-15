# Waste Classification Model

This document provides an overview of the waste classification model developed to identify 30 different types of recyclable and waste items.

## Model Overview

The model uses a **ResNet18** backbone with transfer learning, optimized for detailed waste item classification. It's designed to work well on Apple Silicon devices using the MPS (Metal Performance Shaders) backend.

### Key Features

- **Transfer Learning**: Uses pre-trained ResNet18 as the backbone
- **Custom Classifier**: Added custom classification head optimized for waste types
- **Detailed Classification**: 30 specific waste classes rather than general categories
- **Data Structure**: Handles both default and real-world images in separate subdirectories
- **Hardware Optimization**: Optimized for Apple Silicon GPUs using MPS

## Dataset

The dataset consists of 15,000 images across 30 waste classes:

- **Images per class**: 500 (250 default + 250 real-world)
- **Total dataset size**: 15,000 images
- **Train/Val/Test split**: 70% / 15% / 15% (10,500 / 2,250 / 2,250)

### Class Structure

Each class contains both default images (computer-generated or studio) and real-world images:

```
images/
  ├── aerosol_cans/
  │   ├── default/ (250 images)
  │   └── real_world/ (250 images)
  ├── aluminum_food_cans/
  │   ├── default/ (250 images)
  │   └── real_world/ (250 images)
  └── ... (28 more classes)
```

### Classes

The model classifies 30 specific waste types:

1. aerosol_cans
2. aluminum_food_cans
3. aluminum_soda_cans
4. cardboard_boxes
5. cardboard_packaging
6. clothing
7. coffee_grounds
8. disposable_plastic_cutlery
9. eggshells
10. food_waste
11. glass_beverage_bottles
12. glass_cosmetic_containers
13. glass_food_jars
14. magazines
15. newspaper
16. office_paper
17. paper_cups
18. plastic_cup_lids
19. plastic_detergent_bottles
20. plastic_food_containers
21. plastic_shopping_bags
22. plastic_soda_bottles
23. plastic_straws
24. plastic_trash_bags
25. plastic_water_bottles
26. shoes
27. steel_food_cans
28. styrofoam_cups
29. styrofoam_food_containers
30. tea_bags

## Model Architecture

- **Backbone**: ResNet18 (pre-trained on ImageNet)
- **Custom Classifier**:
  ```
  Sequential(
    (0): Linear(in_features=512, out_features=512)
    (1): ReLU()
    (2): Dropout(p=0.5)
    (3): Linear(in_features=512, out_features=30)
  )
  ```
- **Total Parameters**: 11,454,558 trainable parameters
- **Input Size**: 224×224 RGB images

## Training Configuration

- **Batch Size**: 64
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Optimizer**: AdamW with weight decay (1e-4)
- **Loss Function**: Cross Entropy Loss
- **Epochs**: 20
- **Data Augmentation**:
  - Random horizontal flip (p=0.5)
  - Random vertical flip (p=0.2)
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation)

## Model Performance

Progress tracking includes:
- Per-epoch training and validation loss
- Accuracy and F1 score metrics
- Runtime performance metrics
- Detailed per-class precision, recall, and F1 scores

## Checkpointing

The training process saves several model checkpoints:
- **Best Model (Accuracy)**: Saved whenever validation accuracy improves
- **Best Model (F1)**: Saved whenever F1 score improves (if accuracy is reasonable)
- **Regular Checkpoints**: Saved every 5 epochs
- **Final Model**: Saved at the end of training

## Visualization

The training process generates several visualizations:
- **Training Metrics**: Loss, accuracy, and F1 score over epochs
- **Confusion Matrix**: Detailed matrix showing class predictions
- **Per-class Performance**: Bar charts showing precision, recall, and F1 score for each class

## Inference

A prediction function is included for making predictions on new images:
- Supports loading saved model checkpoints
- Returns top-k predictions with confidence scores
- Handles preprocessing of new images

## Comparison with Similar Models

Compared to models like MobileNetV2-based classifiers:
- **More Detailed Classes**: 30 specific types vs. 6 general categories
- **Higher Parameter Count**: ~11.5M vs. ~3.5M parameters
- **Better Real-World Performance**: Trained on both standard and real-world images
- **Hardware Optimization**: Specifically optimized for Apple Silicon

## Usage

To use the trained model for prediction:

```python
# Load the model
model_path = 'models/best_model_acc.pth'
checkpoint = torch.load(model_path)
model = ImprovedModel(num_classes=len(checkpoint['class_names']))
model.load_state_dict(checkpoint['model_state_dict'])
class_names = checkpoint['class_names']

# Make a prediction
predictions = predict_waste_class('path/to/image.jpg', model, class_names, device)
for i, pred in enumerate(predictions):
    print(f"#{i+1}: {pred['class']} ({pred['confidence']:.2f}%)")
``` 