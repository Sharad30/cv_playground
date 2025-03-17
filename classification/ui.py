import streamlit as st
import torch
import numpy as np
from dataloader import ImageDataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from config import DATA_DIR, BATCH_SIZE, IMG_SIZE

def show_comparison(raw_images, raw_labels, transformed_images, transformed_labels):
    # Create a figure with 4 rows and 2 columns (for raw and transformed pairs)
    fig, axes = plt.subplots(4, 2, figsize=(10, 15))
    
    for idx in range(min(len(raw_images), 4)):
        # Plot raw image
        raw_img = raw_images[idx]
        axes[idx, 0].imshow(raw_img)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f'Raw: {raw_labels[idx]}')
        
        # Plot transformed image
        if torch.is_tensor(transformed_images[idx]):
            trans_img = transformed_images[idx].permute(1, 2, 0).numpy()
            trans_img = trans_img * 0.5 + 0.5  # Reverse the normalization
            trans_img = np.clip(trans_img, 0, 1)
        
        axes[idx, 1].imshow(trans_img)
        axes[idx, 1].axis('off')
        axes[idx, 1].set_title(f'Transformed: {transformed_labels[idx]}')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def main():
    st.title("CIFAR-10 Binary Classification Data Viewer")
    st.write("Visualizing Cat vs Dog Classification Dataset")

    # Load raw CIFAR-10 dataset
    try:
        raw_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return

    # Get indices for cats and dogs
    binary_classes = [3, 5]  # 3 for cat, 5 for dog
    class_names = {3: 'Cat', 5: 'Dog'}
    indices = [i for i, label in enumerate(raw_dataset.targets) if label in binary_classes]

    # Create DataLoader and get the transform
    image_loader = ImageDataLoader(DATA_DIR, BATCH_SIZE)
    data_loader = image_loader.get_data_loader()

    # Display dataset statistics in sidebar
    st.sidebar.header("Dataset Statistics")
    total_cats = sum(1 for i in indices if raw_dataset.targets[i] == 3)
    total_dogs = sum(1 for i in indices if raw_dataset.targets[i] == 5)
    
    st.sidebar.write(f"Total Cats: {total_cats}")
    st.sidebar.write(f"Total Dogs: {total_dogs}")
    st.sidebar.write(f"Total Images: {len(indices)}")

    # Main content
    st.header("Raw vs Transformed Images Comparison")
    
    if st.button("Show New Samples"):
        # Get random samples for raw images
        sample_indices = np.random.choice(indices, size=4, replace=False)
        raw_images = [raw_dataset.data[i] for i in sample_indices]
        raw_labels = [class_names[raw_dataset.targets[i]] for i in sample_indices]

        # Get transformed versions of the same images
        transform = image_loader.transform  # Access transform from ImageDataLoader instance
        transformed_images = []
        for idx in sample_indices:
            img = raw_dataset.data[idx]
            # Convert to PIL Image for transformation
            img = transforms.ToPILImage()(img)
            # Apply transformations
            transformed_img = transform(img)
            transformed_images.append(transformed_img)
        
        transformed_images = torch.stack(transformed_images)
        transformed_labels = raw_labels  # Same labels as raw images

        # Show comparison
        show_comparison(raw_images, raw_labels, 
                       transformed_images, transformed_labels)

        # Display transformation details
        st.write("### Transformation Pipeline:")
        st.write("1. Resize to {}x{}".format(IMG_SIZE[0], IMG_SIZE[1]))
        st.write("2. Random Horizontal Flip")
        st.write("3. Convert to Tensor (0-1 range)")
        st.write("4. Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]")

if __name__ == "__main__":
    main()
