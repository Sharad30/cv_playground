# classification/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import ImageDataLoader
from model import BinaryClassifier
from config import (
    DATA_DIR, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE,
    MODEL_NAME, PRETRAINED, FREEZE_BACKBONE,
    LEARNING_RATE, UNFREEZE_LAYERS
)

def train_model(data_dir=DATA_DIR, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get transforms for training
    train_transforms = ImageDataLoader.get_default_transforms(img_size=IMG_SIZE, train=True)
    
    # Create data loader with custom transforms
    data_loader = ImageDataLoader(
        data_dir, 
        batch_size=batch_size,
        img_size=IMG_SIZE,
        transform=train_transforms
    ).get_data_loader()
    
    # Initialize model with specified configuration
    model = BinaryClassifier(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE
    ).to(device)

    # Print model details
    print(f"\nModel Architecture: {MODEL_NAME}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Initial trainable parameters: {model.get_trainable_params():,}")

    # Unfreeze specified layers if needed
    if UNFREEZE_LAYERS is not None:
        model.unfreeze_layers(UNFREEZE_LAYERS)
        print(f"Unfrozen last {UNFREEZE_LAYERS} layers")
        print(f"Updated trainable parameters: {model.get_trainable_params():,}")
        
    # Print which layers are trainable
    model.print_trainable_layers()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx}/{len(data_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100 * correct/total:.2f}%')

        # Epoch summary
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {epoch_acc:.2f}%\n')

if __name__ == "__main__":
    train_model()