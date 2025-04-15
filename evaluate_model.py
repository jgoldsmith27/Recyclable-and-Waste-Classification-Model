import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import time
import logging
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import multiprocessing

# Try to import necessary classes from the training script
# Assume waste_classification.py is in the same directory or Python path
try:
    from waste_classification import WasteImageDataset, SubsetDataset, ImprovedModel
except ImportError:
    print("Error: Could not import necessary classes from waste_classification.py.")
    print("Please ensure waste_classification.py is in the same directory or accessible in the Python path.")
    exit()

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def evaluate_saved_model(checkpoint_path, data_dir='images'):
    """Evaluates a saved model checkpoint on the test dataset."""
    logger.info(f"Starting evaluation for checkpoint: {checkpoint_path}")
    model_filename_base = os.path.splitext(os.path.basename(checkpoint_path))[0]

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return

    # Load the checkpoint
    try:
        # Load to CPU first for flexibility, in case evaluation device is different
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) 
        class_names = checkpoint['class_names']
        model_state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'N/A') # Get epoch if available
        val_acc_saved = checkpoint.get('val_acc', 'N/A') # Get saved val acc if available
        val_f1_saved = checkpoint.get('val_f1', 'N/A') # Get saved val f1 if available
        logger.info(f"Checkpoint loaded successfully. Contains model from epoch {epoch}.")
        if val_acc_saved != 'N/A':
             logger.info(f"Saved Metrics (from checkpoint): Val Acc={val_acc_saved:.2f}%, Val F1={val_f1_saved:.4f}")
        else:
             logger.info("Saved metrics (val_acc, val_f1) not found in checkpoint.")
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Create model and load weights
    num_classes = len(class_names)
    model = ImprovedModel(num_classes=num_classes)
    try:
        # Load the state dict
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        logger.info("Model state dictionary loaded successfully onto device.")
        
        # Print model parameter count (optional)
        def count_parameters(model):
             return sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {count_parameters(model):,} trainable parameters")

    except Exception as e:
        logger.error(f"Error loading model state dict: {e}")
        return

    # --- Data Loading --- 
    image_size = 224 # Standard size used during training
    batch_size = 64 # Can be adjusted based on evaluation hardware memory

    # Define the same transformations used for validation/testing during training
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to load the test dataset split consistently
    def load_test_data(data_dir='images', transform=None):
        logger.info(f"Loading test dataset split from {data_dir}")
        
        # Instantiate the full dataset structure ONLY to get the correct indices
        # Use the provided test_transform for this instance
        try:
            full_dataset_for_indices = WasteImageDataset(
                root_dir=data_dir, 
                transform=transform,
                include_default=True, # Assume default settings were used for training
                include_real_world=True # Assume default settings were used for training
            )
        except ValueError as e:
             logger.error(f"Failed to initialize dataset: {e}")
             raise # Re-raise the exception to stop execution
        except FileNotFoundError:
             logger.error(f"Dataset directory not found: {data_dir}")
             raise
             
        n_samples = len(full_dataset_for_indices)
        if n_samples == 0:
            raise ValueError("No images found in the dataset directory using current settings.")

        # Use the *exact same seed and split logic* as in the original training script
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        # test_size = n_samples - train_size - val_size # Not strictly needed here

        indices = list(range(n_samples))
        np.random.seed(42) # CRITICAL: Use the same seed as training script (default was 42)
        np.random.shuffle(indices)
        
        test_indices = indices[train_size+val_size:]
        logger.info(f"Derived {len(test_indices)} samples for the test set using the original split logic (seed 42).")
        
        # Create the actual test dataset using the correct indices and transforms
        # We re-use the full_dataset_for_indices object which already has the right transform
        test_dataset = SubsetDataset(full_dataset_for_indices, test_indices)
        
        # Return the test dataset and the class names derived from the dataset structure
        return test_dataset, full_dataset_for_indices.classes

    try:
        test_dataset, loaded_class_names = load_test_data(data_dir=data_dir, transform=test_transforms)
        # Verify class names match between checkpoint and loaded data
        if loaded_class_names != class_names:
             logger.warning(f"Class names mismatch! Checkpoint: {len(class_names)} classes. Dataset: {len(loaded_class_names)} classes. Using names from checkpoint for reporting.")
             # You might want to add stricter error handling here depending on the expected behavior
    except Exception as e:
        logger.error(f"Fatal error loading or splitting test dataset: {e}")
        return # Stop evaluation if data loading fails

    # Create test dataloader
    is_macos = (os.name == 'posix' and 'darwin' in os.sys.platform)
    # Force num_workers to 0 on macOS for stability, otherwise use a reasonable default or os.cpu_count()
    num_workers = 0 if is_macos else min(os.cpu_count(), 4) 
    logger.info(f"Using {num_workers} worker processes for data loading")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Evaluation --- 
    criterion = nn.CrossEntropyLoss() # The loss function used during training

    def evaluate_model_on_test(model, dataloader, criterion, device):
        model.eval() # Set model to evaluation mode
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        logger.info("Running evaluation on the test set...")
        # Use no_grad context manager for efficiency
        with torch.no_grad():
            # Wrap dataloader with tqdm for progress bar
            test_pbar = tqdm(dataloader, desc="Evaluating", leave=False) 
            for images, labels in test_pbar:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate loss and calculate accuracy
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for metrics calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate final metrics
        if test_total == 0: # Handle empty dataset case
             logger.error("Test dataset yielded no samples during evaluation.")
             return 0.0, 0.0, 0.0, [], []

        test_loss /= test_total
        test_acc = 100.0 * test_correct / test_total
        # Use zero_division=0 to handle cases where a class might have no predicted samples
        test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) 
        
        return test_loss, test_acc, test_f1, all_preds, all_labels

    # Run the evaluation
    try:
        test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_model_on_test(model, test_dataloader, criterion, device)
    except Exception as e:
         logger.error(f"An error occurred during model evaluation: {e}")
         return # Stop if evaluation fails

    logger.info(f"--- Evaluation Complete for: {os.path.basename(checkpoint_path)} ---")
    logger.info(f"Test Results -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1 Score (Weighted): {test_f1:.4f}")

    # --- Reporting and Visualization --- 
    if not test_labels or not test_preds:
        logger.warning("No predictions or labels were generated, skipping report and visualizations.")
        return

    # Generate detailed classification report (log and print)
    try:
        # Use zero_division=0 for robustness
        report_dict = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True, zero_division=0)
        report_text = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)
        
        logger.info("--- Detailed Classification Report --- ")
        for cls_name in class_names:
            if cls_name in report_dict:
                 metrics = report_dict[cls_name]
                 logger.info(f"Class '{cls_name}': P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}")
            else:
                 logger.warning(f"Metrics for class '{cls_name}' not found in report (likely zero support).")
        
        # Log overall metrics from the report
        logger.info(f"Overall (macro avg): P={report_dict['macro avg']['precision']:.4f}, R={report_dict['macro avg']['recall']:.4f}, F1={report_dict['macro avg']['f1-score']:.4f}")
        logger.info(f"Overall (weighted avg): P={report_dict['weighted avg']['precision']:.4f}, R={report_dict['weighted avg']['recall']:.4f}, F1={report_dict['weighted avg']['f1-score']:.4f}")
        logger.info(f"Overall Accuracy: {report_dict['accuracy']:.4f}")

        print("\n--- Classification Report ---")
        print(report_text)

    except Exception as e:
        logger.error(f"Error generating classification report: {e}")

    # Define unique filenames for plots based on the model checkpoint
    cm_filename = f"evaluation_confusion_matrix_{model_filename_base}.png"
    metrics_filename = f"evaluation_per_class_metrics_{model_filename_base}.png"

    # Generate and save Confusion Matrix
    try:
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(20, 16)) # Adjust figure size as needed
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8}) # Adjust font size if needed
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_filename_base}')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(cm_filename, dpi=300)
        logger.info(f"Confusion matrix saved as '{cm_filename}'")
        plt.close() # Close the plot to free memory
    except Exception as e:
        logger.error(f"Error generating or saving confusion matrix: {e}")

    # Generate and save Per-class Performance Metrics Plot
    try:
        plt.figure(figsize=(15, 8))
        metrics_to_plot = ['precision', 'recall', 'f1-score']
        num_classes = len(class_names)
        x = np.arange(num_classes)
        width = 0.25 # Width of the bars

        for i, metric in enumerate(metrics_to_plot):
            # Extract values safely, defaulting to 0 if class not in report
            values = [report_dict.get(cls, {}).get(metric, 0) for cls in class_names]
            plt.bar(x + i*width, values, width, label=metric.capitalize())

        plt.xlabel('Waste Classes')
        plt.ylabel('Score')
        plt.title(f'Per-class Performance Metrics - {model_filename_base}')
        plt.xticks(x + width, class_names, rotation=45, ha='right', fontsize=9)
        plt.ylim(0, 1.05) # Set y-axis limits
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(metrics_filename, dpi=300)
        logger.info(f"Per-class metrics saved as '{metrics_filename}'")
        plt.close() # Close the plot
    except Exception as e:
        logger.error(f"Error generating or saving per-class metrics plot: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # Specify the path to the model checkpoint you want to evaluate
    checkpoint_to_evaluate = 'models/continued_best_model_acc.pth'
    # Specify the root directory of your image dataset
    dataset_directory = 'images'
    # ---------------------

    logger.info("========================================================")
    logger.info(f" Starting Evaluation Script ")
    logger.info(f" Evaluating Model: {checkpoint_to_evaluate}")
    logger.info(f" Using Dataset: {dataset_directory}")
    logger.info("========================================================")

    # Fix for multiprocessing issues on macOS if needed (though num_workers=0 bypasses this)
    if os.name == 'posix' and 'darwin' in os.sys.platform:
        try:
            # Check if already set by another module, avoid error if so
            if multiprocessing.get_start_method(allow_none=True) is None:
                 multiprocessing.set_start_method('spawn')
                 logger.info("Set multiprocessing start method to 'spawn' for macOS.")
        except RuntimeError as e:
             logger.warning(f"Could not set multiprocessing start method (may already be set): {e}")

    # Run the evaluation function
    start_eval_time = time.time()
    evaluate_saved_model(checkpoint_path=checkpoint_to_evaluate, data_dir=dataset_directory)
    end_eval_time = time.time()

    logger.info("========================================================")
    logger.info(f" Evaluation script finished in {end_eval_time - start_eval_time:.2f} seconds.")
    logger.info("========================================================") 