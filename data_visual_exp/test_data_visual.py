# test_data_visual.py
"""
Simple test script to run CIFAR-10 data visualization
Run this to check your CIFAR-10 dataset
"""

from data_visual import CIFAR10DataVisualizer

def quick_check():
    """Quick check of CIFAR-10 data without heavy visualizations."""
    print("CIFAR-10 Dataset Quick Check")
    print("=" * 40)
    
    visualizer = CIFAR10DataVisualizer()
    
    # Basic dataset info
    print(f"Train dataset size: {len(visualizer.train_dataset)}")
    print(f"Test dataset size: {len(visualizer.test_dataset)}")
    print(f"Number of classes: {len(visualizer.cifar10_classes)}")
    print(f"Classes: {visualizer.cifar10_classes}")
    
    # Check first image
    image, label = visualizer.train_dataset[0]
    print(f"\nFirst image shape: {image.shape}")
    print(f"First image label: {visualizer.cifar10_classes[label]}")
    print(f"Image data type: {image.dtype}")
    print(f"Pixel value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Check data loader
    print("\nData loader check:")
    batch_images, batch_labels = next(iter(visualizer.data_setup.train_loader))
    print(f"Batch shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Normalized image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")

if __name__ == "__main__":
    quick_check()