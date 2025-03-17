# classification/config.py
DATA_DIR = 'data'
BATCH_SIZE = 32
NUM_EPOCHS = 5
IMG_SIZE = (224, 224)

# Model configuration
MODEL_NAME = 'resnet18'  # Options: 'resnet18', 'efficientnet_b0', 'vit_b_16', etc.
PRETRAINED = True
FREEZE_BACKBONE = True
LEARNING_RATE = 1e-3
UNFREEZE_LAYERS = 2  # Number of layers to unfreeze during training (None for all)