# analyze_depthwise_separable.py
"""
Analysis script for Depthwise Separable Convolutions
Compares parameter count, computation, and efficiency
"""

import torch
import torch.nn as nn
from cifar10model_v0 import Net, DepthwiseSeparableConv

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_depthwise_separable():
    """Analyze Depthwise Separable Convolution benefits."""
    print("Depthwise Separable Convolution Analysis")
    print("=" * 50)
    
    # Compare regular conv vs depthwise separable
    in_channels, out_channels = 32, 64
    kernel_size = 3
    
    # Regular convolution
    regular_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    
    # Depthwise separable convolution
    depthwise_sep = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, padding=1)
    
    # Calculate parameters
    regular_params = count_parameters(regular_conv)
    depthwise_params = count_parameters(depthwise_sep)
    
    print(f"\nParameter Comparison (32→64 channels, 3x3 kernel):")
    print(f"Regular Conv:           {regular_params:,} parameters")
    print(f"Depthwise Separable:    {depthwise_params:,} parameters")
    print(f"Reduction:              {regular_params - depthwise_params:,} parameters")
    print(f"Efficiency:             {depthwise_params/regular_params:.2%} of original")
    print(f"Parameter Savings:      {1 - depthwise_params/regular_params:.2%}")
    
    # Theoretical calculation
    print(f"\nTheoretical Calculation:")
    print(f"Regular Conv parameters:     {out_channels} × {in_channels} × {kernel_size}² = {out_channels * in_channels * kernel_size**2:,}")
    print(f"Depthwise parameters:        {in_channels} × 1 × {kernel_size}² = {in_channels * kernel_size**2:,}")
    print(f"Pointwise parameters:        {out_channels} × {in_channels} × 1² = {out_channels * in_channels:,}")
    print(f"Total Depthwise Sep:         {in_channels * kernel_size**2 + out_channels * in_channels:,}")
    
    # Compute FLOPs estimation
    input_size = 16  # 16x16 feature map
    regular_flops = out_channels * in_channels * kernel_size**2 * input_size**2
    depthwise_flops = (in_channels * kernel_size**2 + out_channels * in_channels) * input_size**2
    
    print(f"\nFLOPs Comparison (16x16 input):")
    print(f"Regular Conv FLOPs:      {regular_flops:,}")
    print(f"Depthwise Sep FLOPs:     {depthwise_flops:,}")
    print(f"FLOP Reduction:          {1 - depthwise_flops/regular_flops:.2%}")

def analyze_full_model():
    """Analyze the full model with depthwise separable convolutions."""
    print(f"\n" + "=" * 50)
    print("Full Model Analysis")
    print("=" * 50)
    
    model = Net()
    total_params = count_parameters(model)
    
    print(f"Total Model Parameters: {total_params:,}")
    
    # Analyze each component
    components = [
        ("Input Conv Block", model.input_conv),
        ("Feature Block (Depthwise)", model.feature_block),
        ("Pattern Block (Mixed)", model.pattern_block),
        ("Classifier", model.classifier)
    ]
    
    print(f"\nParameter Distribution:")
    for name, component in components:
        params = count_parameters(component)
        percentage = (params / total_params) * 100
        print(f"{name:<25}: {params:>6,} params ({percentage:>5.1f}%)")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nModel Output Shape: {output.shape}")
    print(f"Input Shape: {dummy_input.shape}")

def compare_architectures():
    """Compare different architectural choices."""
    print(f"\n" + "=" * 50)
    print("Architecture Comparison")
    print("=" * 50)
    
    # Create different versions for comparison
    
    # Version 1: All regular convolutions
    class RegularNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(),
                
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(),
                
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(),
                
                nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(),
                
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10)
            )
        
        def forward(self, x):
            return self.features(x)
    
    # Version 2: All depthwise separable
    class DepthwiseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(),
                
                DepthwiseSeparableConv(16, 32, stride=2),
                DepthwiseSeparableConv(32, 64),
                DepthwiseSeparableConv(64, 64, stride=2),
                
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10)
            )
        
        def forward(self, x):
            return self.features(x)
    
    regular_net = RegularNet()
    depthwise_net = DepthwiseNet()
    hybrid_net = Net()  # Our hybrid approach
    
    models = [
        ("Regular Conv Only", regular_net),
        ("Depthwise Sep Only", depthwise_net),
        ("Hybrid (Our Model)", hybrid_net)
    ]
    
    print(f"Model Comparison:")
    for name, model in models:
        params = count_parameters(model)
        print(f"{name:<20}: {params:>6,} parameters")

def demonstrate_depthwise_concept():
    """Demonstrate how depthwise separable convolution works."""
    print(f"\n" + "=" * 50)
    print("Depthwise Separable Convolution Concept")
    print("=" * 50)
    
    print("Depthwise Separable Conv = Depthwise Conv + Pointwise Conv")
    print()
    print("1. DEPTHWISE CONVOLUTION:")
    print("   - Each input channel is convolved separately")
    print("   - Uses groups=input_channels in PyTorch")
    print("   - 3x3 kernel → each channel gets its own 3x3 filter")
    print("   - Spatial filtering without channel mixing")
    print()
    print("2. POINTWISE CONVOLUTION:")
    print("   - 1x1 convolution to mix channels")
    print("   - Combines information from depthwise step")
    print("   - Controls output channels")
    print()
    print("Benefits:")
    print("✅ Fewer parameters (typically 8-9x reduction)")
    print("✅ Less computation (similar reduction)")
    print("✅ Often maintains accuracy")
    print("✅ Better for mobile/edge deployment")
    print("✅ Regularization effect")
    print()
    print("Use Cases:")
    print("📱 MobileNets (Google)")
    print("📱 EfficientNets")
    print("📱 Mobile computer vision")
    print("🚀 Real-time applications")

def main():
    """Run all analyses."""
    analyze_depthwise_separable()
    analyze_full_model()
    compare_architectures()
    demonstrate_depthwise_concept()
    
    print(f"\n" + "=" * 50)
    print("Summary: Your model now uses Depthwise Separable Convolutions!")
    print("✅ Reduced parameters and computation")
    print("✅ Maintained expressiveness")
    print("✅ Better efficiency for CIFAR-10")
    print("=" * 50)

if __name__ == "__main__":
    main()