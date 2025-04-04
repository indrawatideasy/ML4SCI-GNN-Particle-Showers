# Proton vs Electron Image Classification with PyTorch

This project focuses on classifying particle image data — specifically **photons** and **electrons** — using convolutional neural networks (CNNs) with PyTorch. The dataset comes in HDF5 format and is used to train a deep learning model for binary classification.

---

## 🧠 Objectives

- Load and preprocess image datasets stored in `.hdf5` format.
- Build an image classifier using transfer learning (ResNet18).
- Normalize image data and split into training/testing sets.
- Train the model using AdamW optimizer and StepLR scheduler.
- Evaluate model performance and address overfitting.

---

## 📁 Dataset

| Particle | Path |
|----------|------|
| Photon   | `/kaggle/input/photon-data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5` |
| Electron | `/kaggle/input/single-electron/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5` |

Each dataset contains image data under the key `"X"` in `(N, H, W, C)` format.

---

## ⚙️ Installation

Install the required Python libraries:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn
pip install h5py
```

---

## 📾 Data Loading and Preprocessing

```python
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_data(photon_path, electron_path):
    with h5py.File(photon_path, "r") as f:
        photons = f["X"][:]
    with h5py.File(electron_path, "r") as f:
        electrons = f["X"][:]

    photon_labels = np.ones(len(photons))
    electron_labels = np.zeros(len(electrons))

    X = np.concatenate((photons, electrons), axis=0).astype(np.float32) / 255.0
    y = np.concatenate((photon_labels, electron_labels), axis=0)

    return X, y

# Load and convert data
X, y = load_data(photon_path, electron_path)
X_tensor = torch.tensor(X).permute(0, 3, 1, 2)  # Shape: (N, C, H, W)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train/test split
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_dataset, test_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

---

## 🧠 Model Architecture (ResNet18)

```python
import torch.nn as nn
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

---

## ⚙️ Training Setup

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Hyperparameters
epochs = 20
learning_rate = 0.0005
weight_decay = 1e-2

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
```

---

## 🚀 Training Loop

```python
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

print("Training complete!")
```

---

## 📊 Results

| Metric           | Value     |
|------------------|-----------|
| Final Train Acc  | **76.17%** |
| Final Test Acc   | **71.51%** |

---

## 🔮 Addressing Overfitting

- **Weight Decay:** Introduced regularization through `weight_decay=1e-2` in AdamW optimizer.
- **Learning Rate Scheduler:** StepLR reduces the learning rate periodically to refine learning.
- **Validation Gap:** A small gap (~4.6%) suggests mild overfitting; can be reduced with:
  - **Data Augmentation** (e.g., rotations, flips)
  - **Dropout Layers**
  - **Early Stopping**
  - **More training data**

---

## 📊 Future Enhancements

- Add metrics like precision, recall, F1-score
- Use Grad-CAM to visualize CNN focus regions
- Visualize learning curves for better diagnosis
- Explore deeper models (ResNet34, ResNet50)
- Apply K-fold cross-validation for robust evaluation

---

## 👨‍💼 Author

**Deasy Indrawati**  
Final-year Information Systems student | Passionate about AI research, machine learning, and natural language processing

---

