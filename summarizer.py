
# summarizer.py
import torch.nn as nn
from torchsummary import summary

# -----------------------------
# 8. Model Architecture Checks
# -----------------------------
def model_checks(model):
    import logging
    
    logging.info('--- Model Architecture Checks ---')
    # Total Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total Parameters: {total_params:,}')
    logging.info(f'Trainable Parameters: {trainable_params:,}\n')
    
    logging.info('Layer-wise Parameter Details (in model order):')
    logging.info('-'*100)
    
    def get_layer_details(module):
        details = ''
        if isinstance(module, nn.Conv2d):
            details = (f'Convolution: {module.in_channels}->{module.out_channels} channels, '
                      f'kernel {module.kernel_size}, stride {module.stride}, padding {module.padding}, '
                      f'groups {module.groups}, bias {module.bias is not None}')
        elif isinstance(module, nn.BatchNorm2d):
            details = f'BatchNorm: {module.num_features} features, eps={module.eps}, momentum={module.momentum}'
        elif isinstance(module, nn.ReLU):
            details = 'Activation: ReLU'
        elif isinstance(module, nn.ReLU6):
            details = 'Activation: ReLU6'
        elif isinstance(module, nn.LeakyReLU):
            details = f'Activation: LeakyReLU (negative_slope={module.negative_slope})'
        elif isinstance(module, nn.MaxPool2d):
            details = f'MaxPool: kernel {module.kernel_size}, stride {module.stride}, padding {module.padding}'
        elif isinstance(module, nn.AvgPool2d):
            details = f'AvgPool: kernel {module.kernel_size}, stride {module.stride}, padding {module.padding}'
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            details = f'AdaptiveAvgPool: output size {module.output_size}'
        elif isinstance(module, nn.Dropout):
            details = f'Dropout: probability {module.p}'
        elif isinstance(module, nn.Dropout2d):
            details = f'Dropout2d: probability {module.p}'
        elif isinstance(module, nn.Linear):
            details = f'Linear: {module.in_features}->{module.out_features}, bias {module.bias is not None}'
        elif isinstance(module, nn.Flatten):
            details = 'Flatten'
        return details

    # Get layers in order as defined in model
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            logging.info(f"\nBlock: {name} (Sequential)")
            for subname, submodule in module.named_children():
                layer_name = f'{name}.{subname} ({submodule.__class__.__name__})'
                layer_params = sum(p.numel() for p in submodule.parameters())
                details = get_layer_details(submodule)
                logging.info(f'  {layer_name:50} | Params: {layer_params:6,d} | {details}')
        else:
            layer_name = f'{name} ({module.__class__.__name__})'
            layer_params = sum(p.numel() for p in module.parameters())
            details = get_layer_details(module)
            logging.info(f'  {layer_name:50} | Params: {layer_params:6,d} | {details}')

    logging.info('-'*100)
    logging.info('\nLayer Type Summary:')
    
    # Count all layer types
    layer_types = {
        'Conv2d': [m for m in model.modules() if isinstance(m, nn.Conv2d)],
        'BatchNorm2d': [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)],
        'ReLU': [m for m in model.modules() if isinstance(m, (nn.ReLU, nn.ReLU6))],
        'LeakyReLU': [m for m in model.modules() if isinstance(m, nn.LeakyReLU)],
        'MaxPool2d': [m for m in model.modules() if isinstance(m, nn.MaxPool2d)],
        'AvgPool2d': [m for m in model.modules() if isinstance(m, nn.AvgPool2d)],
        'AdaptiveAvgPool2d': [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)],
        'Dropout': [m for m in model.modules() if isinstance(m, nn.Dropout)],
        'Dropout2d': [m for m in model.modules() if isinstance(m, nn.Dropout2d)],
        'Linear': [m for m in model.modules() if isinstance(m, nn.Linear)],
        'Flatten': [m for m in model.modules() if isinstance(m, nn.Flatten)]
    }
    
    for layer_type, layers in layer_types.items():
        if layers:  # Only show if there are layers of this type
            logging.info(f'{layer_type:20} layers used: {len(layers):3d}')
    
    logging.info('-'*100)


