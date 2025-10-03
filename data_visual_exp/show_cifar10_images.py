# show_cifar10_images.py
"""
Simple script to display CIFAR-10 images using matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

def show_cifar10_samples():
    """Display sample CIFAR-10 images in a grid."""
    
    # CIFAR-10 class names
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Load CIFAR-10 dataset without normalization for display
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    
    # Create a 4x4 grid of images
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('CIFAR-10 Dataset - Sample Images', fontsize=16, fontweight='bold')
    
    # Get 16 random samples
    indices = np.random.choice(len(train_dataset), 16, replace=False)
    
    for i, idx in enumerate(indices):
        image, label = train_dataset[idx]
        
        # Convert tensor to numpy array and transpose for matplotlib
        # PyTorch tensors are (C, H, W), matplotlib expects (H, W, C)
        image_np = image.permute(1, 2, 0).numpy()
        
        # Calculate subplot position
        row = i // 4
        col = i % 4
        
        # Display image
        axes[row, col].imshow(image_np)
        axes[row, col].set_title(f'{cifar10_classes[label]}', fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Dataset info:")
    print(f"- Total training images: {len(train_dataset)}")
    print(f"- Image shape: {train_dataset[0][0].shape}")
    print(f"- Number of classes: {len(cifar10_classes)}")

def show_class_examples():
    """Show one example from each CIFAR-10 class."""
    
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    
    # Find first example of each class
    class_examples = {}
    for i, (image, label) in enumerate(train_dataset):
        if label not in class_examples:
            class_examples[label] = image
        if len(class_examples) == 10:  # Found all classes
            break
    
    # Create 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('CIFAR-10 Classes - One Example Each', fontsize=16, fontweight='bold')
    
    for class_idx in range(10):
        image = class_examples[class_idx]
        image_np = image.permute(1, 2, 0).numpy()
        
        row = class_idx // 5
        col = class_idx % 5
        
        axes[row, col].imshow(image_np)
        axes[row, col].set_title(f'{cifar10_classes[class_idx]}', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_image_statistics():
    """Show pixel value distribution for CIFAR-10."""
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    
    # Sample some images for analysis
    sample_size = 1000
    indices = np.random.choice(len(train_dataset), sample_size, replace=False)
    
    all_pixels_r = []
    all_pixels_g = []
    all_pixels_b = []
    
    print("Analyzing pixel statistics...")
    for idx in indices:
        image, _ = train_dataset[idx]
        all_pixels_r.extend(image[0].flatten().tolist())
        all_pixels_g.extend(image[1].flatten().tolist())
        all_pixels_b.extend(image[2].flatten().tolist())
    
    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('CIFAR-10 Pixel Value Distributions', fontsize=14, fontweight='bold')
    
    axes[0].hist(all_pixels_r, bins=50, alpha=0.7, color='red', label='Red Channel')
    axes[0].set_title('Red Channel')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(all_pixels_g, bins=50, alpha=0.7, color='green', label='Green Channel')
    axes[1].set_title('Green Channel')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    
    axes[2].hist(all_pixels_b, bins=50, alpha=0.7, color='blue', label='Blue Channel')
    axes[2].set_title('Blue Channel')
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nPixel Statistics (from {sample_size} images):")
    print(f"Red   - Mean: {np.mean(all_pixels_r):.4f}, Std: {np.std(all_pixels_r):.4f}")
    print(f"Green - Mean: {np.mean(all_pixels_g):.4f}, Std: {np.std(all_pixels_g):.4f}")
    print(f"Blue  - Mean: {np.mean(all_pixels_b):.4f}, Std: {np.std(all_pixels_b):.4f}")

def main():
    """Run all visualization functions."""
    print("CIFAR-10 Image Visualization with Matplotlib")
    print("=" * 50)
    
    try:
        print("\n1. Showing random sample images...")
        show_cifar10_samples()
        
        print("\n2. Showing class examples...")
        show_class_examples()
        
        print("\n3. Showing pixel statistics...")
        show_image_statistics()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have matplotlib, torch, and torchvision installed.")

if __name__ == "__main__":
    main()