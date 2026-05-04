import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNNTransformerModel
from dataset import CustomImageDataset
from utils import final_loss

# Hyperparameters for training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100

# 1. Use the proper supervised dataset
# This loads paired GT (ground truth) and NOISY images from the SIDD dataset
print("Loading dataset...")
train_dataset = CustomImageDataset('d:/saveetha/SIDD_Small_sRGB_Only/Data')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"Dataset loaded. Total batches: {len(train_loader)}")

# Initialize Model
model = CNNTransformerModel()

# Loss and Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")
model = model.to(device)

# Training Loop
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    for noisy, clean in train_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        outputs = model(noisy)
        
        # Calculate loss (MSE + SSIM)
        loss = final_loss(outputs, clean, noisy)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss

print("Starting training...")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    avg_loss = train(model, train_loader, optimizer)
    scheduler.step()

# Save the model
model = model.cpu()
torch.save(model.state_dict(), 'enhancement_model_fixed.pth')
print("Model saved successfully as enhancement_model_fixed.pth!")
