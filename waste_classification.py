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
from sklearn.metrics import f1_score, accuracy_score, classification_report
import time
import logging
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import multiprocessing

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'training_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create a directory for saving models
os.makedirs('models', exist_ok=True)

"""## Custom Dataset for Loading Images from Nested Folders (default/real_world)"""
class WasteImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_default=True, include_real_world=True):
        self.root_dir = root_dir
        self.transform = transform
        self.include_default = include_default
        self.include_real_world = include_real_world
        
        if not include_default and not include_real_world:
            raise ValueError("At least one of include_default or include_real_world must be True")
        
        # Get the waste type classes (top-level directories)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Track statistics for logging
        class_stats = {cls: {'default': 0, 'real_world': 0} for cls in self.classes}
        
        # Collect all image paths and their labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            
            # Check for default and real_world subdirectories
            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)
                
                # Only process directories
                if not os.path.isdir(subdir_path):
                    continue
                    
                # Skip directories based on settings
                if subdir == 'default' and not include_default:
                    continue
                if subdir == 'real_world' and not include_real_world:
                    continue
                
                # Only process default and real_world directories
                if subdir not in ['default', 'real_world']:
                    logger.warning(f"Unexpected subdirectory in {class_dir}: {subdir}")
                    continue
                
                # Process images in this subdirectory
                for img_name in os.listdir(subdir_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        self.image_paths.append(os.path.join(subdir_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])
                        class_stats[class_name][subdir] += 1
        
        # Log detailed statistics
        for class_name in self.classes:
            default_count = class_stats[class_name]['default']
            real_world_count = class_stats[class_name]['real_world']
            total_count = default_count + real_world_count
            logger.info(f"Class '{class_name}': {total_count} images (default: {default_count}, real_world: {real_world_count})")
        
        # Log summary
        total_default = sum(stats['default'] for stats in class_stats.values())
        total_real_world = sum(stats['real_world'] for stats in class_stats.values())
        total_images = len(self.image_paths)
        
        logger.info(f"Loaded {total_images} images from {len(self.classes)} classes")
        logger.info(f"Default images: {total_default}, Real-world images: {total_real_world}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define SubsetDataset outside the function for multiprocessing
class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

"""## CNN Model with ResNet Backbone for Transfer Learning"""
class ImprovedModel(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()
        # Use ResNet18 as backbone for transfer learning
        self.backbone = models.resnet18(weights='DEFAULT' if use_pretrained else None)
        in_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def run_training():
    """## Training and Testing"""
    # Set the hyperparameters
    batch_size = 64  # Larger batch size for better GPU utilization
    num_epochs = 20
    learning_rate = 0.001
    image_size = 224  # Standard size for many CNN models

    # Data transformations with more augmentation for improved generalization
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data transformations for validation/test (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def load_and_split_data(data_dir='images', include_default=True, include_real_world=True):
        logger.info(f"Loading dataset from {data_dir}")
        logger.info(f"Including default images: {include_default}, Including real-world images: {include_real_world}")
        
        # Load the full dataset
        full_dataset = WasteImageDataset(
            root_dir=data_dir, 
            transform=data_transforms,
            include_default=include_default,
            include_real_world=include_real_world
        )
        
        # Get total number of samples
        n_samples = len(full_dataset)
        
        if n_samples == 0:
            raise ValueError("No images found with the current settings. Please check directory paths and inclusion settings.")
        
        # Generate train/val/test indices
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - val_size
        
        # Split indices
        indices = list(range(n_samples))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        logger.info(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
        
        # Create a separate dataset for validation and test to apply different transforms
        val_dataset = WasteImageDataset(
            root_dir=data_dir, 
            transform=val_transforms,
            include_default=include_default,
            include_real_world=include_real_world
        )
        test_dataset = WasteImageDataset(
            root_dir=data_dir, 
            transform=val_transforms,
            include_default=include_default,
            include_real_world=include_real_world
        )
        
        # Create subsets using the SubsetDataset class defined outside the function
        train_dataset = SubsetDataset(full_dataset, train_indices)
        val_dataset = SubsetDataset(val_dataset, val_indices)
        test_dataset = SubsetDataset(test_dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset, full_dataset.classes

    # By default, use both default and real_world images
    # You can adjust these flags to use only specific subsets (e.g., training on default, testing on real_world)
    include_default = True
    include_real_world = True

    train_dataset, val_dataset, test_dataset, class_names = load_and_split_data(
        include_default=include_default,
        include_real_world=include_real_world
    )

    # Disable multiprocessing on macOS to avoid issues
    # This is slower but more stable on macOS
    is_macos = (os.name == 'posix' and 'darwin' in os.sys.platform)
    num_workers = 0 if is_macos else min(os.cpu_count(), 4) 
    logger.info(f"Using {num_workers} worker processes for data loading")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Print sample batch shape for debugging
    for images, labels in train_dataloader:
        logger.info(f"Sample batch shape: {images.shape}, Labels shape: {labels.shape}")
        break

    """##Build Model"""
    # Use MPS (Metal Performance Shaders) on M1/M2 Macs if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Create the model, loss function, and optimizer
    num_classes = len(class_names)
    logger.info(f"Creating model with {num_classes} classes")
    model = ImprovedModel(num_classes=num_classes)
    model = model.to(device)

    # Print model summary and total parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model has {count_parameters(model):,} trainable parameters")

    # Using weighted cross entropy loss if classes are imbalanced
    criterion = nn.CrossEntropyLoss()

    # Use AdamW optimizer with weight decay to reduce overfitting
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    """##Train Model"""
    def evaluate_model(model, dataloader, criterion, device):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(dataloader.dataset)
        val_acc = 100.0 * val_correct / val_total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return val_loss, val_acc, f1, all_preds, all_labels

    # Lists to store the training and validation metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []

    # Best validation accuracy for model checkpointing
    best_val_acc = 0.0
    best_val_f1 = 0.0

    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= len(train_dataset)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Log epoch results
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f}s - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}, LR: {current_lr:.6f}")
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': class_names
            }, os.path.join('models', 'best_model_acc.pth'))
            logger.info(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
        
        # Also save a model version that has the best F1 score (may be different from best accuracy)
        if val_f1 > best_val_f1 and val_acc > best_val_acc * 0.95:  # Only if accuracy is also reasonable
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': class_names
            }, os.path.join('models', 'best_model_f1.pth'))
            logger.info(f"Saved new best model with validation F1: {val_f1:.4f}")
        
        # Save a checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': class_names
            }, os.path.join('models', f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"Saved checkpoint at epoch {epoch+1}")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes!")

    # Save the final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_accs[-1],
        'val_f1': val_f1s[-1],
        'class_names': class_names
    }, os.path.join('models', 'final_model.pth'))
    logger.info("Final model saved!")

    """##Testing Model with Best Checkpoint"""
    # Load the best model for testing
    logger.info("Loading best model for testing...")
    best_model_path = os.path.join('models', 'best_model_acc.pth')
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']} with validation accuracy {best_checkpoint['val_acc']:.2f}%")

    # Test the model
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_model(model, test_dataloader, criterion, device)

    logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1 Score: {test_f1:.4f}")

    # Generate detailed classification report
    report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    logger.info("Classification Report:")
    for cls in class_names:
        logger.info(f"Class '{cls}': Precision={report[cls]['precision']:.4f}, Recall={report[cls]['recall']:.4f}, F1={report[cls]['f1-score']:.4f}, Support={report[cls]['support']}")

    # Generate standard report for console
    print_report = classification_report(test_labels, test_preds, target_names=class_names)
    print("Classification Report:")
    print(print_report)

    # Plot training and validation metrics
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, 'b-', label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs+1), val_f1s, 'g-', label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)

    # Add learning rate subplot if we tracked it
    current_lr = optimizer.param_groups[0]['lr']
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f"Test Accuracy: {test_acc:.2f}%\nTest F1 Score: {test_f1:.4f}\nFinal LR: {current_lr:.6f}",
            horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    logger.info("Training metrics saved as 'training_metrics.png'")

    """## Confusion Matrix Visualization"""
    from sklearn.metrics import confusion_matrix

    # Create confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    logger.info("Confusion matrix saved as 'confusion_matrix.png'")

    # Plot per-class performance metrics
    plt.figure(figsize=(15, 8))
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in class_names]
        plt.bar(x + i*width, values, width, label=metric)

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-class Performance Metrics')
    plt.xticks(x + width, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('per_class_metrics.png', dpi=300)
    logger.info("Per-class metrics saved as 'per_class_metrics.png'")

    return model, class_names, device

"""## Inference Function for New Images"""
def predict_waste_class(image_path, model, class_names, device, top_k=3):
    # Load the model if it's a path
    if isinstance(model, str):
        checkpoint = torch.load(model)
        loaded_model = ImprovedModel(num_classes=len(class_names))
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        model = loaded_model
    
    model = model.to(device)
    model.eval()
    
    # Process the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        topk_probs, topk_indices = torch.topk(probs, k=min(top_k, len(class_names)))
        
    # Return top predictions with their probabilities
    predictions = []
    for i in range(len(topk_indices)):
        idx = topk_indices[i].item()
        predictions.append({
            'class': class_names[idx],
            'confidence': topk_probs[i].item() * 100
        })
    
    return predictions


if __name__ == "__main__":
    # Fix for multiprocessing on macOS
    if os.name == 'posix':
        multiprocessing.set_start_method('spawn')
    
    # Run the training process
    model, class_names, device = run_training()
    
    logger.info("Script completed successfully!")
    
    # Example of using the model for prediction (commented out)
    """
    test_image = 'path/to/test/image.jpg'
    predictions = predict_waste_class(test_image, model, class_names, device)
    for i, pred in enumerate(predictions):
        print(f"#{i+1}: {pred['class']} ({pred['confidence']:.2f}%)")
    """