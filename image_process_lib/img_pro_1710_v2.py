import torch
import torch.nn.functional as F

class image_processing:
    def __init__(self):
        self.red = 0
        self.green = 1
        self.blue = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gaussian_blur(self, image_tensor, kernel_size=3, sigma=1.0):
        """Apply Gaussian blur to reduce Gaussian noise by smoothing the image.
        - Uses a Gaussian kernel to average pixel values, weighted by distance.
        - Effective for additive noise but may blur edges if kernel_size or sigma is too large.
        - Parameters:
            - image_tensor: Input tensor [B, 3, H, W] or [3, H, W], values in [0, 255].
            - kernel_size: Size of the Gaussian kernel (odd integer, e.g., 3 or 5).
            - sigma: Standard deviation of the Gaussian distribution (controls blur strength).
        - Returns: Denoised tensor of same shape, values in [0, 255].
        """
        original_shape = image_tensor.shape
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim: [1, 3, H, W]
        
        # Ensure kernel_size is odd
        kernel_size = int(kernel_size) | 1
        
        # Create Gaussian kernel manually
        coords = torch.arange(kernel_size, dtype=torch.float32, device=image_tensor.device)
        coords = coords - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Create 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(image_tensor.size(1), 1, kernel_size, kernel_size).contiguous()
        
        # Apply convolution
        padding = kernel_size // 2
        denoised = F.conv2d(image_tensor, kernel, padding=padding, groups=image_tensor.size(1))
        
        # Restore original shape
        if len(original_shape) == 3:
            denoised = denoised.squeeze(0)  # Remove batch dim: [3, H, W]
        
        return denoised.clamp(0, 255)

    def median_filter(self, image_tensor, kernel_size=3):
        """Apply median filter to remove salt-and-pepper noise while preserving edges.
        - Replaces each pixel with the median of its neighborhood.
        - Good for impulse noise; less blurring than Gaussian for small kernels.
        - Parameters:
            - image_tensor: Input tensor [B, 3, H, W] or [3, H, W], values in [0, 255].
            - kernel_size: Size of the neighborhood (odd integer, e.g., 3 or 5).
        - Returns: Denoised tensor of same shape, values in [0, 255].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim: [1, 3, H, W]
        
        # Ensure kernel_size is odd
        kernel_size = int(kernel_size) | 1
        # Pad the image to handle edges
        padding = kernel_size // 2
        padded = F.pad(image_tensor, (padding, padding, padding, padding), mode='reflect')
        
        # Unfold to get all neighborhoods
        unfolded = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # [B, C, H, W, k, k]
        # Compute median over the last two dimensions (kernel)
        denoised = unfolded.median(dim=-1)[0].median(dim=-1)[0]  # [B, C, H, W]
        
        if image_tensor.dim() == 3:
            denoised = denoised.squeeze(0)  # Remove batch dim: [3, H, W]
        
        return denoised.clamp(0, 255)

    def bilateral_filter(self, image_tensor, kernel_size=5, sigma_spatial=1.5, sigma_color=30.0):
        """Apply bilateral filter to smooth noise while preserving edges.
        - Weights pixels by spatial distance and intensity difference, reducing noise without blurring edges.
        - Suitable for Gaussian or natural noise in CIFAR-10.
        - Parameters:
            - image_tensor: Input tensor [B, 3, H, W] or [3, H, W], values in [0, 255].
            - kernel_size: Size of the neighborhood (odd integer, e.g., 5).
            - sigma_spatial: Standard deviation for spatial Gaussian (controls spatial weighting).
            - sigma_color: Standard deviation for intensity Gaussian (controls intensity weighting).
        - Returns: Denoised tensor of same shape, values in [0, 255].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim: [1, 3, H, W]
        
        # Ensure kernel_size is odd
        kernel_size = int(kernel_size) | 1
        padding = kernel_size // 2
        
        # Create spatial Gaussian kernel
        x = torch.arange(-padding, padding + 1, device=self.device).float()
        y = x.view(-1, 1)
        spatial = torch.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))  # [k, k]
        spatial = spatial / spatial.sum()  # Normalize
        spatial = spatial.view(1, 1, kernel_size, kernel_size)
        
        # Pad image
        padded = F.pad(image_tensor, (padding, padding, padding, padding), mode='reflect')
        
        # Apply bilateral filter per channel
        denoised = torch.zeros_like(image_tensor)
        for c in range(image_tensor.shape[1]):  # Process each channel
            unfolded = padded[:, c:c+1].unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # [B, 1, H, W, k, k]
            diff = unfolded - image_tensor[:, c:c+1, :, :, None, None]  # Intensity difference
            color_weights = torch.exp(-(diff**2) / (2 * sigma_color**2))  # [B, 1, H, W, k, k]
            weights = color_weights * spatial  # Combine spatial and color weights
            weights = weights / weights.sum(dim=(-2, -1), keepdim=True)  # Normalize
            denoised[:, c:c+1] = (unfolded * weights).sum(dim=(-2, -1))
        
        if image_tensor.dim() == 3:
            denoised = denoised.squeeze(0)  # Remove batch dim: [3, H, W]
        
        return denoised.clamp(0, 255)

    def total_variation_denoising(self, image_tensor, weight=0.1, max_iter=50):
        """Apply total variation denoising to remove noise while preserving edges.
        - Minimizes total variation (sum of gradient magnitudes) using gradient descent.
        - Effective for piecewise-smooth images; good for structured noise.
        - Parameters:
            - image_tensor: Input tensor [B, 3, H, W] or [3, H, W], values in [0, 255].
            - weight: Regularization parameter for TV term (higher = smoother image).
            - max_iter: Number of optimization iterations.
        - Returns: Denoised tensor of same shape, values in [0, 255].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim: [1, 3, H, W]
        
        # Initialize denoised image as a copy of input (requires grad for optimization)
        denoised = image_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([denoised], lr=0.1)
        
        for _ in range(max_iter):
            optimizer.zero_grad()
            # Data fidelity term: ||denoised - input||^2
            data_term = F.mse_loss(denoised, image_tensor)
            # TV term: sum of gradient magnitudes
            dx = denoised[:, :, 1:, :] - denoised[:, :, :-1, :]
            dy = denoised[:, :, :, 1:] - denoised[:, :, :, :-1]
            tv_term = (dx.abs().sum() + dy.abs().sum()) / (image_tensor.shape[0] * image_tensor.shape[1])
            # Total loss
            loss = data_term + weight * tv_term
            loss.backward()
            optimizer.step()
        
        if image_tensor.dim() == 3:
            denoised = denoised.squeeze(0)  # Remove batch dim: [3, H, W]
        
        return denoised.detach().clamp(0, 255)

    def norm_image_to_red_green_blue_channels(self, image_tensor):
        """Normalize image tensor to [0, 1] and split into RGB channels."""
        # Ensure input is float and normalized to [0, 1]
        img_norm = image_tensor.float() / 255.0
        # Split into RGB channels (assuming shape [B, C, H, W] or [C, H, W])
        red = img_norm[:, self.red:self.red+1, :, :]  # Keep channel dim
        green = img_norm[:, self.green:self.green+1, :, :]
        blue = img_norm[:, self.blue:self.blue+1, :, :]
        return red, green, blue

    def get_pixel_value_mean_std(self, channel, kernel=1):
        """Compute mean and std of surrounding pixels using convolution, excluding padded values."""
        # Ensure channel is [B, 1, H, W]
        if channel.dim() == 3:
            channel = channel.unsqueeze(0)  # Add batch dim if needed

        # Create a uniform kernel for computing local mean (3x3 or larger based on kernel)
        kernel_size = 2 * kernel + 1
        kernel_weights = torch.ones(1, 1, kernel_size, kernel_size, device=channel.device)
        # Set center of kernel to 0 to exclude the pixel itself
        kernel_weights[:, :, kernel, kernel] = 0
        # Compute the number of valid neighbors (excluding center)
        num_neighbors = kernel_weights.sum().item()

        # Create a mask to count valid (non-padded) neighbors for each output pixel
        ones = torch.ones_like(channel)  # [B, 1, H, W]
        valid_counts = F.conv2d(ones, kernel_weights, padding=kernel)
        # Avoid division by zero by clamping minimum valid counts
        valid_counts = torch.clamp(valid_counts, min=1e-8)

        # Compute local mean using convolution
        mean = F.conv2d(channel, kernel_weights, padding=kernel) / valid_counts

        # Compute local std: std = sqrt(E[X^2] - (E[X])^2)
        squared = channel ** 2
        mean_squared = F.conv2d(squared, kernel_weights, padding=kernel) / valid_counts
        variance = mean_squared - (mean ** 2)
        # Clamp variance to avoid negative values due to numerical errors
        variance = torch.clamp(variance, min=1e-8)
        std = torch.sqrt(variance)

        # Compute output: (pixel - mean) / std
        output = (channel - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
        return output

    def get_max_channel_index(self, rgb_image):
        """Get index of the channel with maximum value."""
        # rgb_image: [B, 3, H, W]
        max_channel_index = torch.argmax(rgb_image, dim=1, keepdim=True)  # [B, 1, H, W]
        return max_channel_index

    def get_processed_image(self, image_tensor, kernel=1):
        """Process image to compute normalized channels."""
        # Move tensor to device
        image_tensor = image_tensor.to(self.device)
        
        # Normalize and split into RGB channels
        red, green, blue = self.norm_image_to_red_green_blue_channels(image_tensor)
        
        # Process each channel
        processed_red = self.get_pixel_value_mean_std(red, kernel)
        processed_green = self.get_pixel_value_mean_std(green, kernel)
        processed_blue = self.get_pixel_value_mean_std(blue, kernel)
        
        # Stack processed channels: [B, 3, H, W]
        processed_image = torch.cat([processed_red, processed_green, processed_blue], dim=1)
        return processed_image

    def output_image_array(self, image_tensor, processed_image):
        """Extract and amplify pixel values based on max channel index and its mean-variance difference."""
        # Move inputs to device
        image_tensor = image_tensor.to(self.device)
        processed_image = processed_image.to(self.device)
    
        # Get max channel index: [B, 1, H, W]
        max_channel_index = self.get_max_channel_index(processed_image)
    
        # Gather pixel values from original image based on max channel index
        selected_pixel = torch.gather(image_tensor, dim=1, index=max_channel_index)  # [B, 1, H, W]
    
        # Gather the corresponding processed value (z-score)
        selected_z = torch.gather(processed_image, dim=1, index=max_channel_index)  # [B, 1, H, W]
    
        # Amplify based on z-score: amplify only positive deviations
        amplification_factor = 2 + torch.clamp(selected_z, min=0.0)
        output = selected_pixel * amplification_factor
    
        # Clip to [0, 255]
        output = torch.clamp(output, min=0.0, max=255.0)
    
        return output  # [B, 1, H, W]

    def extract_image_features(self, input_tensor, kernel=1, denoise_method=None, denoise_params=None, include_processed_channel=True):
        """Main method to process batched or single images, with optional denoising and processed channel.
        - Parameters:
            - input_tensor: Input tensor [B, 3, H, W] or [3, H, W], values in [0, 255].
            - kernel: Kernel size for z-score computation.
            - denoise_method: String ('gaussian', 'median', 'bilateral', 'tvd') or None to skip denoising.
            - denoise_params: Dict of parameters for the chosen denoising method (e.g., {'kernel_size': 3, 'sigma': 1.0}).
            - include_processed_channel: Boolean, if True, includes processed channel (output [B, 4, H, W]); if False, returns only RGB ([B, 3, H, W]).
        - Returns: Tensor [B, 4, H, W] or [B, 3, H, W] depending on include_processed_channel.
        """
        # Handle both batched and single inputs
        input_was_3d = input_tensor.dim() == 3
        if input_was_3d:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dim: [1, 3, H, W]
        
        # Apply denoising if specified
        if denoise_method is not None:
            denoise_params = denoise_params or {}
            if denoise_method == 'gaussian':
                input_tensor = self.gaussian_blur(input_tensor, **denoise_params)
            elif denoise_method == 'median':
                input_tensor = self.median_filter(input_tensor, **denoise_params)
            elif denoise_method == 'bilateral':
                input_tensor = self.bilateral_filter(input_tensor, **denoise_params)
            elif denoise_method == 'tvd':
                input_tensor = self.total_variation_denoising(input_tensor, **denoise_params)
            else:
                raise ValueError(f"Unknown denoising method: {denoise_method}")
        
        # If not including processed channel, return the (possibly denoised) input
        if not include_processed_channel:
            if input_was_3d:
                input_tensor = input_tensor.squeeze(0)  # Remove batch dim: [3, H, W]
            return input_tensor
        
        # Process image to compute normalized channels
        processed_image = self.get_processed_image(input_tensor, kernel=kernel)
        processed_channel = self.output_image_array(input_tensor, processed_image)  # [B, 1, H, W]
        
        # Concatenate original RGB channels with processed channel
        final_image = torch.cat([input_tensor, processed_channel], dim=1)  # [B, 4, H, W]
        
        # Remove batch dim if input was single image
        if input_was_3d:
            final_image = final_image.squeeze(0)  # [4, H, W]
        
        return final_image