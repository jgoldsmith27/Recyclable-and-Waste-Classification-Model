import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from waste_classification_model import CNN, ComplexCNN, ImprovedModel


# Check if dataset files exist in the current directory
if os.path.exists('X_dataset.npy') and os.path.exists('Y_dataset.npy'):
    file_path_X = 'X_dataset.npy'
    file_path_Y = 'Y_dataset.npy'
    print("Loading datasets from local files...")
    X_dataset = np.load(file_path_X)
    Y_dataset = np.load(file_path_Y)
    
    # Print data shapes for debugging
    print(f"X_dataset shape: {X_dataset.shape}")
    print(f"Y_dataset shape: {Y_dataset.shape}")
    
    # Check if we need to permute dimensions - if data is in NHWC format, convert to NCHW
    if len(X_dataset.shape) == 4 and X_dataset.shape[3] == 3:
        print("Converting from NHWC to NCHW format...")
        X_dataset = np.transpose(X_dataset, (0, 3, 1, 2))
        print(f"New X_dataset shape: {X_dataset.shape}")
        
    # Reshape Y_dataset if needed to 1D
    if len(Y_dataset.shape) > 1:
        Y_dataset = Y_dataset.squeeze()
        print(f"Squeezed Y_dataset shape: {Y_dataset.shape}")
else:
    print("Dataset files not found. Please ensure X_dataset.npy and Y_dataset.npy are in the current directory.")
    exit(1)


"""## Training and Testing"""
# Set the hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.0005

def load_and_split(X_dataset, y_dataset):
    # Split the dataset into training, validation, and test sets
    x_train, x_val_test, y_train, y_val_test = train_test_split(X_dataset, y_dataset, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=1/3, random_state=42)

    # Create tensor datasets
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long())
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val).long())
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long())

    return train_dataset, val_dataset, test_dataset


def main():
    train_dataset, val_dataset, test_dataset = load_and_split(X_dataset, Y_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print sample batch shape for debugging
    for images, labels in train_dataloader:
        print(f"Sample batch shape: {images.shape}, Labels shape: {labels.shape}")
        break

    """##Build Model"""
    # Create the model, loss function, and optimizer
    num_classes = 6
    # model = CNN(num_classes)
    # model = ComplexCNN(num_classes)
    # model = ResNet()
    model = ImprovedModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("checking model trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")

    # Check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    """##Train Model"""

    # Lists to store the training and validation losses
    train_losses = []
    val_losses = []

    # Lists to store the training and validation losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training completed!")

    # Save the model
    torch.save(model.state_dict(), 'waste_classification_model.pth')
    print("Model saved!")

    # Optional: Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    print("Loss curve saved as 'loss_curve.png'")

if __name__ == "__main__":
    main()