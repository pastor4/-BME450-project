# Title
"Early Detection of Brain Tumors Using MRI and Deep Learning"

## Team Members
Anthony Pastor (pastor4)

## Project Description
I aim to build a deep learning model to classify brain tumors from T1-weighted contrast-enhanced MRI images. The model will be trained to distinguish between three common tumor types: glioma, meningioma, and pituitary tumors, as well as scans with no tumors present. I plan on using publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which contains over 3,000 MRI images labeled by tumor type, already classified into testing and training sets. The goal is to preprocess the images, train a CNN model using PyTorch, and evaluate performance with metrics such as accuracy, precision, and recall. The model should be able to accurately screen an individual for a brain tumor and determine which (if any) type of brain tumor is detected. If there is time, I'd like to try and map where specifically on the MRI scan the model found each tumor (or where it falsly labeled one).

# ===============================================
# 📦 STEP 0 – Install and Import Required Libraries
# ===============================================
# This cell installs PyTorch Lightning (if not preinstalled),
# and imports all necessary libraries for:
# - Deep learning (PyTorch, torchvision, PyTorch Lightning)
# - Data loading and preprocessing
# - Model evaluation and visualization
# - Google Colab file handling (upload/download)

!pip install pytorch-lightning

# Core Python & OS
import os
import zipfile

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# torchvision (for image datasets and transformations)
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Colab-specific (for uploading files)
from google.colab import files

# Optional (for model evaluation and image plotting)
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ===========================================================
# 📂 STEP 1 – Upload and Extract Dataset from ZIP File
# ===========================================================
# This cell:
# - Uploads the "BME 450 Final Dataset.zip" file via Colab's file dialog
# - Automatically extracts the dataset contents into /content/
# - Searches for "Training" and "Testing" folders
# - Sets the paths for later use in dataset loading
# - Verifies that the correct folders and class directories were found

from google.colab import files
import zipfile, os

# Upload zip file
uploaded = files.upload()  # Upload your "BME 450 Final Dataset.zip"

# Extract contents
zip_filename = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('/content')

# Display folder contents for troubleshooting
# print("Top-level contents of /content:")
# print(os.listdir('/content'))

# Automatically find train and test folders
all_dirs = os.listdir('/content')
if 'Training' in all_dirs and 'Testing' in all_dirs:
    train_path = '/content/Training'
    test_path = '/content/Testing'
    print(f"\n✅ Found Training and Testing folders directly in /content")
else:
    # Look inside a subfolder
    for folder in all_dirs:
        if os.path.isdir(f'/content/{folder}'):
            subdirs = os.listdir(f'/content/{folder}')
            if 'Training' in subdirs and 'Testing' in subdirs:
                train_path = f'/content/{folder}/Training'
                test_path = f'/content/{folder}/Testing'
                print(f"\n✅ Found Training and Testing in /content/{folder}")
                break
    else:
        raise FileNotFoundError("❌ Could not locate Training and Testing folders.")

# Final check
print("\nTrain path:", train_path)
print("Test path:", test_path)
print("Train classes:", os.listdir(train_path))

# =============================================================
# 🧼 STEP 2 – Define Image Transforms, Load Dataset, and Create DataLoaders
# =============================================================
# This cell:
# - Defines preprocessing and augmentation steps using torchvision.transforms
# - Loads the MRI image data from the extracted "Training" and "Testing" folders
#   using torchvision.datasets.ImageFolder
# - Wraps the datasets in DataLoaders for efficient mini-batch training
# - Prints class names and sample counts for verification

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_dataset = ImageFolder(root=train_path, transform=transform)
val_dataset   = ImageFolder(root=test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print("Classes:", train_dataset.classes)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# =============================================================
# 🧠 STEP 3 – Define Brain Tumor Classifier Using ResNet18 (Transfer Learning)
# =============================================================
# This cell:
# - Loads a pretrained ResNet18 model from torchvision with ImageNet weights
# - Replaces the final fully connected layer to output 4 tumor classes
# - Wraps the model in a PyTorch LightningModule for modular training
# - Implements training and validation steps with accuracy and loss logging
# - Configures the Adam optimizer with a learning rate of 1e-4

import torchvision.models as models

class BrainTumorClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 4)  # Replace final layer
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# =============================================================
# 🏋️ STEP 4 – Train the Model with EarlyStopping, Checkpointing, and CSV Logging
# =============================================================
# This cell:
# - Instantiates the BrainTumorClassifier model
# - Uses PyTorch Lightning's Trainer with GPU acceleration
# - Logs metrics (loss and accuracy) to a CSV file for later plotting
# - Applies EarlyStopping and saves the best model based on validation loss

from pytorch_lightning.loggers import CSVLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BrainTumorClassifier()

# Set up CSV logger
csv_logger = CSVLogger("logs", name="brain_tumor_model")

trainer = Trainer(
    max_epochs=10,
    accelerator='gpu' if device == 'cuda' else 'cpu',
    devices=1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    ],
    logger=csv_logger,  # ✅ Logger added
    log_every_n_steps=10
)

trainer.fit(model, train_loader, val_loader)

# =============================================================
# 📈 STEP 4.5 – Plot Training and Validation Curves from CSV Logs (Line Graph Version)
# =============================================================
# This cell:
# - Loads training metrics (loss and accuracy) from the CSV logger
# - Plots training and validation loss over time using smooth line graphs
# - Optionally plots training and validation accuracy if available
# - Adapts automatically to the metric columns in your version of PyTorch Lightning

import pandas as pd
import matplotlib.pyplot as plt

# Load metrics CSV logged by PyTorch Lightning
metrics_path = csv_logger.experiment.metrics_file_path
metrics_df = pd.read_csv(metrics_path)

# Show available columns for verification
print("✅ Logged columns:", list(metrics_df.columns))

# Filter rows that contain validation loss (i.e., one per epoch)
metrics_df = metrics_df.dropna(subset=["val_loss"])

# ---- Plot LOSS ----
loss_cols = [col for col in metrics_df.columns if "loss" in col]

plt.figure(figsize=(10, 4))
for col in loss_cols:
    if "epoch" not in col:
        plt.plot(metrics_df["epoch"].values, metrics_df[col].values, marker='o', label=col)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plot ACCURACY (if available) ----
acc_cols = [col for col in metrics_df.columns if "acc" in col]
if acc_cols:
    plt.figure(figsize=(10, 4))
    for col in acc_cols:
        if "epoch" not in col:
            plt.plot(metrics_df["epoch"].values, metrics_df[col].values, marker='o', label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No accuracy columns found in metrics log. Only loss will be plotted.")

# =============================================================
# 💾 STEP 5 – Load Best Model Checkpoint (Based on Validation Loss)
# =============================================================
# This cell:
# - Retrieves the file path to the best model checkpoint saved dur

best_model_path = trainer.checkpoint_callback.best_model_path
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['state_dict'])

# =============================================================
# 🔍 STEP 6 – Visualize Sample Predictions from the Validation Set
# =============================================================
# This cell:
# - Switches the model to evaluation mode and ensures it is on the correct device
# - Retrieves a single batch of validation images and makes predictions
# - Unnormalizes and displays 6 sample images with their predicted and true labels
# - Prints batch-level accuracy as a quick sanity check on performance

import matplotlib.pyplot as plt

model.to(device)  # ✅ FIX: Ensure model is on the same device as the input
model.eval()

images, labels = next(iter(val_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

# Move data to CPU for visualization
images = images.cpu()
preds = preds.cpu()
labels = labels.cpu()
class_names = train_dataset.classes

# Display predictions
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    img = images[i].permute(1, 2, 0) * 0.5 + 0.5
    plt.imshow(img.numpy())
    plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Accuracy on this batch
correct = (preds == labels).sum().item()
total = labels.size(0)
print(f"✅ Sample batch accuracy: {correct}/{total} = {100 * correct / total:.2f}%")

# =============================================================
# 📉 STEP 7 – Generate and Plot Confusion Matrix
# =============================================================
# This cell:
# - Performs inference over the entire validation set without gradient tracking
# - Collects all true and predicted labels
# - Computes a confusion matrix using scikit-learn
# - Uses seaborn to visualize the confusion matrix as a heatmap
# - Helps identify misclassification patterns between tumor types

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Ensure model is in evaluation mode and on correct device
model.to(device)
model.eval()

all_preds = []
all_labels = []

# Run through the full validation set
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
class_names = train_dataset.classes

# Plot using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix – Validation Set')
plt.tight_layout()
plt.show()

# =============================================================
# ✅ STEP 8 – Compute Full Validation Set Accuracy
# =============================================================
# This cell:
# - Iterates through the entire validation set in batches (batch_size=64)
# - Runs inference using the best-trained model in evaluation mode
# - Counts the number of correct predictions vs. total predictions
# - Computes and prints the overall classification accuracy
# - Provides a more reliable metric than single-batch evaluation

from tqdm import tqdm  # Optional: for nice progress bar

model.to(device)
model.eval()

correct = 0
total = 0

# You can increase batch size temporarily here (e.g., 64 or 128 if your GPU can handle it)
eval_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    for images, labels in tqdm(eval_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Full Validation Set Accuracy: {correct}/{total} = {accuracy:.2f}%")

# =============================================================
# 📊 STEP 9 – Generate Classification Report (Precision, Recall, F1-Score)
# =============================================================
# This cell:
# - Performs inference on the entire validation set using the best model
# - Collects all true and predicted labels
# - Uses scikit-learn to compute precision, recall, and F1-score for each class
# - Prints a detailed classification report to evaluate per-class performance
# - Complements the confusion matrix with exact class-wise metrics

from sklearn.metrics import classification_report

# Rerun inference
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print report
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# =============================================================
# 💾 STEP 10 – Save and Download Trained Model
# =============================================================
# This cell:
# - Saves the trained model's state_dict to a .pt file for future use
# - Provides an option to download the model directly to your local machine
# - Ensures your trained model is not lost if the Colab session ends

# Save to file
torch.save(model.state_dict(), "brain_tumor_model.pt")
# Download to your computer (optional)
from google.colab import files
files.download("brain_tumor_model.pt")

