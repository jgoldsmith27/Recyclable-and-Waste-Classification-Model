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
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from waste_classification import X_dataset, Y_dataset, load_and_split
from waste_classification_model import CNN, ComplexCNN, ImprovedModel

def main():
    """##Testing Model"""
    num_classes = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedModel(num_classes)  # or ComplexCNN / CNN if that's what you trained
    model.load_state_dict(torch.load('waste_classification_model.pth'))
    model.to(device)
    model.eval()


    # Set the hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0005
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_dataset, val_dataset, test_dataset = load_and_split(X_dataset, Y_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_dataset)
    accuracy = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")


    # generating testing results in confusion matrix and accuracy

    # Class names for your dataset
    class_names = ['Plastic', 'Paper', 'Glass', 'Metal', 'Organic', 'Textiles']

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Accuracy per class
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))

    for true_label, pred_label in zip(all_labels, all_preds):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1

    class_accuracy = 100 * class_correct / class_total

    # Plotting accuracy per class
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, class_accuracy, color='skyblue')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Class")
    plt.title("Accuracy per Class")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("accuracy_per_class.png")
    plt.close()

if __name__ == "__main__":
    main()
