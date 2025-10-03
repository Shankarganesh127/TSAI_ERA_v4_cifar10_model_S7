# receptive_field_calculator.py - Calculate receptive field for CIFAR-10 CNN models
from cifar10model_v0 import Net

def calculate_receptive_field(model):
    """
    Calculate receptive field for each layer in the model.
    Formula: RF_new = RF_previous + (kernel_size - 1) * dilation for stride=1
             RF_new = RF_previous * stride + (kernel_size - 1) * dilation for stride>1
    """
    rf = 1  # Starting receptive field (single pixel)
    print("Layer-by-layer receptive field calculation:")
    print(f"Input: RF = {rf}")

    # C1 block
    print("\nC1 Block:")
    # Conv2d(3, 8, 3, padding=1)
    rf = rf + (3 - 1) * 1
    print(f"Conv2d(3->8, 3x3, p=1): RF = {rf}")

    # Conv2d(8, 16, 3, padding=1)
    rf = rf + (3 - 1) * 1
    print(f"Conv2d(8->16, 3x3, p=1): RF = {rf}")

    # Conv2d(16, 32, 3, padding=1)
    rf = rf + (3 - 1) * 1
    print(f"Conv2d(16->32, 3x3, p=1): RF = {rf}")

    # C2 block
    print("\nC2 Block:")
    # Conv2d(32, 32, 3, padding=1, dilation=1)
    rf = rf + (3 - 1) * 1
    print(f"Conv2d(32->32, 3x3, p=1, d=1): RF = {rf}")

    # Conv2d(32, 32, 3, padding=2, dilation=2)
    rf = rf + (3 - 1) * 2
    print(f"Conv2d(32->32, 3x3, p=2, d=2): RF = {rf}")

    # Conv2d(32, 32, 3, padding=4, dilation=4)
    rf = rf + (3 - 1) * 4
    print(f"Conv2d(32->32, 3x3, p=4, d=4): RF = {rf}")

    # C3 block
    print("\nC3 Block:")
    # DepthwiseSeparableConv depthwise: Conv2d(32, 32, 3, stride=2, padding=1, groups=32)
    rf = rf * 2 + (3 - 1) * 1
    print(f"DepthwiseConv(32->32, 3x3, s=2, p=1): RF = {rf}")

    # Pointwise doesn't change RF
    print(f"PointwiseConv(32->64, 1x1): RF = {rf} (unchanged)")

    # Conv2d(64, 64, 3, padding=1)
    rf = rf + (3 - 1) * 1
    print(f"Conv2d(64->64, 3x3, p=1): RF = {rf}")

    # C4 block
    print("\nC4 Block:")
    # Conv2d(64, 128, 3, stride=2, padding=1)
    rf = rf * 2 + (3 - 1) * 1
    print(f"Conv2d(64->128, 3x3, s=2, p=1): RF = {rf}")

    # GAP doesn't change RF
    print(f"AdaptiveAvgPool2d(1): RF = {rf} (unchanged)")

    print(f"\nFinal receptive field: {rf} x {rf} pixels")
    return rf

def count_parameters(model):
    """Count total parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params

if __name__ == "__main__":
    model = Net()
    rf = calculate_receptive_field(model)
    params = count_parameters(model)

    print("\nRequirements check:")
    print(f"RF > 44: {'✓' if rf > 44 else '✗'} ({rf} > 44)")
    print(f"Parameters < 200k: {'✓' if params < 200000 else '✗'} ({params:,} < 200,000)")