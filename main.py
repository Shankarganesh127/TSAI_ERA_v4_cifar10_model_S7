
# main.py
import logging
from logger_setup import setup_logging
import summarizer
import io
import contextlib
import train_test
import cifar10model_v0
import sys
import os

# Add image processing library to path
sys.path.append('image_process_lib/')
import cifar10model_v0_impro


class get_model:
    def __init__(self, device=None, use_image_processing=False, use_4_channels=True, 
                 denoising_method=None, denoising_params=None):
        self.device = device if device else self.get_device()
        self.use_image_processing = use_image_processing
        self.use_4_channels = use_4_channels
        self.denoising_method = denoising_method
        self.denoising_params = denoising_params or {}
        self.model_obj = self.get_model()
        self.model_config = self.get_config()

    def get_device(self):
        return train_test.torch.device("cuda" if train_test.torch.cuda.is_available() else "cpu")

    def get_model(self):
        if self.use_image_processing:
            # Use the enhanced model with image processing
            model = cifar10model_v0_impro.Net(
                use_4_channels=self.use_4_channels
            ).to(self.device)
            # Configure denoising if specified
            if self.denoising_method:
                model.denoising_method = self.denoising_method
                model.denoising_params = self.denoising_params
                logging.info(f"Configured denoising: {self.denoising_method} with params: {self.denoising_params}")
            return model
        else:
            # Use the original model without image processing
            return cifar10model_v0.Net().to(self.device)

    def get_config(self):
        if self.use_image_processing:
            return cifar10model_v0_impro.set_config_v0().setup(self.model_obj)
        else:
            return cifar10model_v0.set_config_v0().setup(self.model_obj)

def get_user_preferences():
    """Get user preferences for model configuration"""
    print("\n" + "="*60)
    print("CIFAR-10 Model Configuration")
    print("="*60)
    
    # Choose image processing library
    print("\nAvailable Image Processing Options:")
    print("1. Original Model (cifar10model_v0.py) - Standard 3-channel RGB")
    print("2. Enhanced Model (cifar10model_v0_impro.py) - With image processing")
    
    while True:
        try:
            choice = int(input("\nSelect model type (1 or 2): "))
            if choice in [1, 2]:
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")
    
    use_image_processing = (choice == 2)
    use_4_channels = True  # Default for enhanced model
    denoising_method = None
    denoising_params = {}
    
    if use_image_processing:
        # Ask about channel configuration
        print("\nChannel Configuration:")
        print("1. 3-channel RGB (denoised if denoising selected)")
        print("2. 4-channel RGB + processed/difference channel")
        
        while True:
            try:
                channel_choice = int(input("\nSelect channel configuration (1 or 2): "))
                if channel_choice in [1, 2]:
                    break
                else:
                    print("Please enter 1 or 2")
            except ValueError:
                print("Please enter a valid number")
        
        use_4_channels = (channel_choice == 2)
        
        print("\nAvailable Denoising Methods:")
        print("1. None - No denoising")
        print("2. Gaussian Blur - Smooth noise reduction")
        print("3. Median Filter - Salt-and-pepper noise removal")
        print("4. Bilateral Filter - Edge-preserving smoothing")
        print("5. Total Variation Denoising - Piecewise-smooth denoising")
        
        while True:
            try:
                denoise_choice = int(input("\nSelect denoising method (1-5): "))
                if denoise_choice in range(1, 6):
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        if denoise_choice == 2:
            denoising_method = 'gaussian'
            try:
                kernel_size = int(input("Enter kernel size (odd number, default 3): ") or "3")
                sigma = float(input("Enter sigma value (default 1.0): ") or "1.0")
            except EOFError:
                # Use defaults when input is piped
                kernel_size = 3
                sigma = 1.0
            denoising_params = {'kernel_size': kernel_size, 'sigma': sigma}
        elif denoise_choice == 3:
            denoising_method = 'median'
            try:
                kernel_size = int(input("Enter kernel size (odd number, default 3): ") or "3")
            except EOFError:
                kernel_size = 3
            denoising_params = {'kernel_size': kernel_size}
        elif denoise_choice == 4:
            denoising_method = 'bilateral'
            try:
                kernel_size = int(input("Enter kernel size (odd number, default 5): ") or "5")
                sigma_spatial = float(input("Enter spatial sigma (default 1.5): ") or "1.5")
                sigma_color = float(input("Enter color sigma (default 30.0): ") or "30.0")
            except EOFError:
                kernel_size = 5
                sigma_spatial = 1.5
                sigma_color = 30.0
            denoising_params = {
                'kernel_size': kernel_size, 
                'sigma_spatial': sigma_spatial, 
                'sigma_color': sigma_color
            }
        elif denoise_choice == 5:
            denoising_method = 'tvd'
            try:
                weight = float(input("Enter regularization weight (default 0.1): ") or "0.1")
                max_iter = int(input("Enter max iterations (default 50): ") or "50")
            except EOFError:
                weight = 0.1
                max_iter = 50
            denoising_params = {'weight': weight, 'max_iter': max_iter}
    
    return use_image_processing, use_4_channels, denoising_method, denoising_params

def main_i(params_check=1, use_image_processing=False, use_4_channels=True, denoising_method=None, denoising_params=None):
    logging.info("Setting up for model")
    logging.info(f"Using image processing: {use_image_processing}")
    logging.info(f"Using 4 channels: {use_4_channels}")
    if denoising_method:
        logging.info(f"Denoising method: {denoising_method} with params: {denoising_params}")
    
    model = get_model(device=None, 
                     use_image_processing=use_image_processing, 
                     use_4_channels=use_4_channels,
                     denoising_method=denoising_method, 
                     denoising_params=denoising_params)
    
    # Capture printed summary into logs
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        if use_image_processing:
            summarizer.summary(model.model_obj, input_size=(3, 32, 32))  # Input is still 3-channel RGB
        else:
            summarizer.summary(model.model_obj, input_size=(3, 32, 32))
        summary_text = buf.getvalue().strip()
    if summary_text:
        logging.info("\n" + summary_text)
    
    train_test_instance = train_test.train_test_model(model.model_obj,
                                                      model.device, 
                                                      model.model_config.data_setup_instance.train_loader,
                                                      model.model_config.data_setup_instance.test_loader,
                                                      model.model_config.criterion,
                                                      model.model_config.optimizer,
                                                      model.model_config.scheduler,
                                                      model.model_config.epochs)
    if (params_check == 0):
        train_test_instance.run_epoch()
    else:
        pass
    #train_test_instance.plot_results()
    # Capture printed model checks into logs
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        summarizer.model_checks(model.model_obj)
        checks_text = buf.getvalue().strip()
    if checks_text:
        logging.info("\n" + checks_text)

def main():
    # Initialize logging only in the main process
    setup_logging(log_to_file=True)
    
    # Get user preferences
    use_image_processing, use_4_channels, denoising_method, denoising_params = get_user_preferences()
    
    try:
        params_check = int(input("\nEnter 1 for params check only, 0 for full training/testing: "))
    except EOFError:
        # Default to params check when input is piped
        params_check = 1
        print("Using default: params check only")
    
    main_i(params_check=params_check, 
           use_image_processing=use_image_processing,
           use_4_channels=use_4_channels,
           denoising_method=denoising_method,
           denoising_params=denoising_params)

if __name__ == "__main__":
    main()
