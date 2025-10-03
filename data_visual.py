# data_visual.py
"""
Generic Data Visualization Class for PyTorch Vision Datasets
Supports MNIST, CIFAR-10, CIFAR-100, Fashion-MNIST, and custom datasets
"""
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from collections import Counter

class data_visual:
    """
    Generic data visualization class for any PyTorch vision dataset.
    
    Usage:
        visualizer = data_visual('CIFAR10')
        visualizer.show_sample_images()
        visualizer.show_class_distribution()
    """
    
    def __init__(self, dataset_name='CIFAR10', data_path='../data'):
        """
        Initialize the data visualizer.
        
        Args:
            dataset_name (str): Name of the dataset ('CIFAR10', 'MNIST', 'CIFAR100', 'FashionMNIST')
            data_path (str): Path to store/load dataset
        """
        self.dataset_name = dataset_name.upper()
        self.data_path = data_path
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        
        # Basic transform for visualization (no normalization)
        self.basic_transform = transforms.Compose([transforms.ToTensor()])
        
        # Load datasets
        self.train_dataset, self.test_dataset = self._load_datasets()
        
        print(f"Initialized {self.dataset_name} visualizer")
        print(f"Classes: {self.num_classes}, Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
    
    def _get_class_names(self):
        """Get class names for different datasets."""
        class_mapping = {
            'CIFAR10': [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ],
            'CIFAR100': [f'class_{i}' for i in range(100)],  # CIFAR-100 has 100 classes
            'MNIST': [str(i) for i in range(10)],  # Digits 0-9
            'FASHIONMNIST': [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ]
        }
        
        return class_mapping.get(self.dataset_name, [f'class_{i}' for i in range(10)])
    
    def _load_datasets(self):
        """Load the specified dataset."""
        dataset_classes = {
            'CIFAR10': datasets.CIFAR10,
            'CIFAR100': datasets.CIFAR100,
            'MNIST': datasets.MNIST,
            'FASHIONMNIST': datasets.FashionMNIST
        }
        
        if self.dataset_name not in dataset_classes:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        dataset_class = dataset_classes[self.dataset_name]
        
        print(f"Loading {self.dataset_name} dataset...")
        
        train_dataset = dataset_class(
            self.data_path, train=True, download=True, transform=self.basic_transform
        )
        test_dataset = dataset_class(
            self.data_path, train=False, download=True, transform=self.basic_transform
        )
        
        return train_dataset, test_dataset
    
    def _prepare_image_for_display(self, image_tensor):
        """Prepare image tensor for matplotlib display."""
        if len(image_tensor.shape) == 3:  # Color image (C, H, W)
            if image_tensor.shape[0] == 3:  # RGB
                return image_tensor.permute(1, 2, 0).numpy()
            elif image_tensor.shape[0] == 1:  # Grayscale
                return image_tensor.squeeze(0).numpy()
        elif len(image_tensor.shape) == 2:  # Already 2D grayscale
            return image_tensor.numpy()
        
        return image_tensor.numpy()
    
    def show_sample_images(self, num_samples=16, dataset_type='train', grid_size=None):
        """
        Display sample images from the dataset.
        
        Args:
            num_samples (int): Number of images to display
            dataset_type (str): 'train' or 'test'
            grid_size (tuple): (rows, cols) for custom grid size
        """
        dataset = self.train_dataset if dataset_type == 'train' else self.test_dataset
        
        # Auto-calculate grid size if not provided
        if grid_size is None:
            rows = int(np.sqrt(num_samples))
            cols = int(np.ceil(num_samples / rows))
        else:
            rows, cols = grid_size
            num_samples = min(num_samples, rows * cols)
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        fig.suptitle(f'{self.dataset_name} {dataset_type.capitalize()} Dataset - Sample Images', 
                     fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            image_np = self._prepare_image_for_display(image)
            
            # Display image
            if len(image_np.shape) == 2:  # Grayscale
                axes[i].imshow(image_np, cmap='gray')
            else:  # Color
                axes[i].imshow(image_np)
            
            axes[i].set_title(f'{self.class_names[label]}', fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print dataset info
        sample_image, _ = dataset[0]
        print("\nDataset Info:")
        print(f"- Dataset: {self.dataset_name}")
        print(f"- {dataset_type.capitalize()} images: {len(dataset)}")
        print(f"- Image shape: {sample_image.shape}")
        print(f"- Number of classes: {self.num_classes}")
    
    def show_class_examples(self, samples_per_class=5):
        """
        Show sample images for each class.
        
        Args:
            samples_per_class (int): Number of examples per class
        """
        # Collect samples for each class
        class_samples = {i: [] for i in range(self.num_classes)}
        
        for idx, (image, label) in enumerate(self.train_dataset):
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append((image, idx))
            
            # Break if we have enough samples for all classes
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
        
        # Create figure
        fig, axes = plt.subplots(self.num_classes, samples_per_class, 
                                figsize=(2*samples_per_class, 2*self.num_classes))
        fig.suptitle(f'{self.dataset_name} Classes - Sample Images', fontsize=16, fontweight='bold')
        
        # Handle different subplot configurations
        if self.num_classes == 1:
            axes = [axes] if samples_per_class == 1 else axes
        elif samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        # Display samples
        for class_idx in range(self.num_classes):
            for sample_idx in range(samples_per_class):
                ax = axes[class_idx, sample_idx] if self.num_classes > 1 else axes[sample_idx]
                
                if sample_idx < len(class_samples[class_idx]):
                    image, _ = class_samples[class_idx][sample_idx]
                    image_np = self._prepare_image_for_display(image)
                    
                    if len(image_np.shape) == 2:  # Grayscale
                        ax.imshow(image_np, cmap='gray')
                    else:  # Color
                        ax.imshow(image_np)
                    
                    if sample_idx == 0:  # Label only first column
                        ax.set_ylabel(f'{self.class_names[class_idx]}', 
                                     fontsize=10, rotation=0, labelpad=50)
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def show_class_distribution(self):
        """Show the distribution of classes in train and test datasets."""
        # Get labels for train and test
        train_labels = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        test_labels = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
        
        train_counts = Counter(train_labels)
        test_counts = Counter(test_labels)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{self.dataset_name} Class Distribution', fontsize=16, fontweight='bold')
        
        # Prepare data
        classes = [self.class_names[i] for i in range(self.num_classes)]
        train_values = [train_counts[i] for i in range(self.num_classes)]
        test_values = [test_counts[i] for i in range(self.num_classes)]
        
        # Train distribution
        ax1.bar(range(self.num_classes), train_values, color='skyblue', alpha=0.7)
        ax1.set_title('Train Set Class Distribution')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(self.num_classes))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        
        # Test distribution
        ax2.bar(range(self.num_classes), test_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Test Set Class Distribution')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks(range(self.num_classes))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n{self.dataset_name} Statistics:")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Image shape: {self.train_dataset[0][0].shape}")
    
    def show_pixel_statistics(self, sample_size=1000):
        """
        Analyze and visualize pixel value statistics.
        
        Args:
            sample_size (int): Number of images to analyze
        """
        # Sample images for analysis
        indices = np.random.choice(len(self.train_dataset), 
                                 min(sample_size, len(self.train_dataset)), replace=False)
        
        sample_image, _ = self.train_dataset[0]
        is_color = len(sample_image.shape) == 3 and sample_image.shape[0] == 3
        
        if is_color:
            all_pixels_r, all_pixels_g, all_pixels_b = [], [], []
        else:
            all_pixels = []
        
        print(f"Analyzing pixel statistics from {len(indices)} images...")
        
        for idx in indices:
            image, _ = self.train_dataset[idx]
            
            if is_color:
                all_pixels_r.extend(image[0].flatten().tolist())
                all_pixels_g.extend(image[1].flatten().tolist())
                all_pixels_b.extend(image[2].flatten().tolist())
            else:
                if len(image.shape) == 3:  # (1, H, W)
                    image = image.squeeze(0)
                all_pixels.extend(image.flatten().tolist())
        
        # Create plots
        if is_color:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f'{self.dataset_name} Pixel Value Distributions', fontsize=14, fontweight='bold')
            
            axes[0].hist(all_pixels_r, bins=50, alpha=0.7, color='red')
            axes[0].set_title('Red Channel')
            axes[0].set_xlabel('Pixel Value')
            axes[0].set_ylabel('Frequency')
            
            axes[1].hist(all_pixels_g, bins=50, alpha=0.7, color='green')
            axes[1].set_title('Green Channel')
            axes[1].set_xlabel('Pixel Value')
            axes[1].set_ylabel('Frequency')
            
            axes[2].hist(all_pixels_b, bins=50, alpha=0.7, color='blue')
            axes[2].set_title('Blue Channel')
            axes[2].set_xlabel('Pixel Value')
            axes[2].set_ylabel('Frequency')
            
            # Print statistics
            print(f"\nPixel Statistics (from {len(indices)} images):")
            print(f"Red   - Mean: {np.mean(all_pixels_r):.4f}, Std: {np.std(all_pixels_r):.4f}")
            print(f"Green - Mean: {np.mean(all_pixels_g):.4f}, Std: {np.std(all_pixels_g):.4f}")
            print(f"Blue  - Mean: {np.mean(all_pixels_b):.4f}, Std: {np.std(all_pixels_b):.4f}")
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            fig.suptitle(f'{self.dataset_name} Pixel Value Distribution', fontsize=14, fontweight='bold')
            
            ax.hist(all_pixels, bins=50, alpha=0.7, color='gray')
            ax.set_title('Pixel Values')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            
            # Print statistics
            print(f"\nPixel Statistics (from {len(indices)} images):")
            print(f"Mean: {np.mean(all_pixels):.4f}, Std: {np.std(all_pixels):.4f}")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_all(self):
        """Run all visualization methods."""
        print(f"\n{self.dataset_name} Complete Visualization")
        print("=" * 50)
        
        try:
            print("\n1. Class Distribution...")
            self.show_class_distribution()
            
            print("\n2. Sample Images...")
            self.show_sample_images(16)
            
            print("\n3. Class Examples...")
            samples_per_class = 5 if self.num_classes <= 10 else 3
            self.show_class_examples(samples_per_class)
            
            print("\n4. Pixel Statistics...")
            self.show_pixel_statistics()
            
        except Exception as e:
            print(f"Error during visualization: {e}")

# Convenience functions for quick usage
def visualize_dataset(dataset_name='CIFAR10', data_path='../data'):
    """
    Quick function to visualize any dataset.
    
    Args:
        dataset_name (str): 'CIFAR10', 'MNIST', 'CIFAR100', 'FashionMNIST'
        data_path (str): Path to store/load dataset
    """
    visualizer = data_visual(dataset_name, data_path)
    visualizer.visualize_all()
    return visualizer

def main():
    """Main function demonstrating usage."""
    print("Generic Dataset Visualizer")
    print("=" * 40)
    
    # Example usage
    datasets_to_try = ['CIFAR10', 'MNIST', 'FashionMNIST']
    
    for dataset in datasets_to_try:
        print(f"\nTrying {dataset}...")
        try:
            visualizer = data_visual(dataset)
            visualizer.show_sample_images(9, grid_size=(3, 3))
            break  # Show only first available dataset
        except Exception as e:
            print(f"Failed to load {dataset}: {e}")

if __name__ == "__main__":
    main()
