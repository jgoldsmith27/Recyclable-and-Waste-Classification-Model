## Waste Classification Model

Presentation Slides linked [here](https://docs.google.com/presentation/d/1Ms8EEEM6BegDRybc0nQp-yT8ufclPX8v7us3Wn3yYUM/edit?slide=id.g352744a9fc9_0_57#slide=id.g352744a9fc9_0_57)

This repository contains a pipeline for training, evaluating, and experimenting with a waste classification model using image data. The project includes dataset exploration, training scripts with tunable hyperparameters, and model testing.

## 🚀 Getting Started

### 1. Explore the Dataset

Begin by running the `exploring-the-dataset.ipynb` notebook. This will:

- Download the processed waste image dataset.
- Provide basic visualizations and statistics to help you understand the data.

### 2. Train the Model

To train a model, run the training script from the terminal:

```bash
python waste_classification_train.py
```
This will:
- Automatically fetch the dataset if not already present.
- Train the model using the default architecture and hyperparameters.

#### You can customize the training by:
- Changing the hyperparameters (learning rate, batch size, etc.) inside waste_classification_train.py.
- Swapping the model architecture by modifying the import from the waste_classification_models/ directory.

### 3. Test the Model
After training, evaluate your model's performance:

```bash
python waste_classification_test.py
```
This will load the latest trained model and report test metrics such as accuracy, loss, F1 and downloaded relevant plots such as the confusion matrix, accuracy per class, and validation vs. loss plot.
