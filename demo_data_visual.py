# demo_data_visual.py
"""
Demonstration of the generic data_visual class
Shows how to visualize different datasets
"""

from data_visual import data_visual, visualize_dataset

def demo_single_dataset():
    """Demo using the data_visual class for a single dataset."""
    print("=== Demo: Single Dataset Visualization ===")
    
    # Create visualizer for CIFAR-10
    viz = data_visual('CIFAR10')
    
    # Show different visualizations
    print("\n1. Sample Images (3x3 grid):")
    viz.show_sample_images(9, grid_size=(3, 3))
    
    print("\n2. Class Distribution:")
    viz.show_class_distribution()
    
    print("\n3. Class Examples (3 per class):")
    viz.show_class_examples(3)

def demo_multiple_datasets():
    """Demo comparing different datasets."""
    print("\n=== Demo: Multiple Dataset Comparison ===")
    
    datasets = ['CIFAR10', 'MNIST', 'FashionMNIST']
    
    for dataset_name in datasets:
        print(f"\n--- {dataset_name} ---")
        try:
            viz = data_visual(dataset_name)
            viz.show_sample_images(4, grid_size=(2, 2))
        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")

def demo_convenience_function():
    """Demo using the convenience function."""
    print("\n=== Demo: Convenience Function ===")
    
    # Quick visualization of any dataset
    viz = visualize_dataset('CIFAR10')
    
    return viz

def demo_custom_usage():
    """Demo custom usage patterns."""
    print("\n=== Demo: Custom Usage Patterns ===")
    
    # CIFAR-10 analysis
    viz = data_visual('CIFAR10')
    
    # Custom sample display
    print("\n1. Large sample grid:")
    viz.show_sample_images(25, grid_size=(5, 5))
    
    # Pixel analysis
    print("\n2. Pixel statistics:")
    viz.show_pixel_statistics(500)  # Analyze 500 images

if __name__ == "__main__":
    print("Generic Data Visualizer Demo")
    print("=" * 50)
    
    # Run demos
    demo_single_dataset()
    #demo_multiple_datasets()
    # demo_convenience_function()  # Uncomment for full visualization
    #demo_custom_usage()