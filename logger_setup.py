# logger_setup.py
import logging
import os

_LOGGING_INITIALIZED = False

class TqdmLoggingHandler(logging.StreamHandler):
    """A logging handler that plays nicely with tqdm progress bars."""
    def emit(self, record):
        try:
            from tqdm import tqdm  # Lazy import so tqdm isn't a hard dependency
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            # Fallback to normal stream behavior
            super().emit(record)

def setup_logging(log_to_file=False, log_dir='logs'):
    """Set up simple logging configuration.
    
    Args:
        log_to_file (bool): If True, also log to a file (default: False)
        log_dir (str): Directory for log files if log_to_file is True
    """
    global _LOGGING_INITIALIZED

    # If already initialized, don't recreate handlers (prevents truncation mid-run)
    if _LOGGING_INITIALIZED:
        logger = logging.getLogger()
        # If file logging is requested and not present yet, add it (append mode)
        if log_to_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            try:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, 'training.log')
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                file_handler.setLevel(logging.INFO)
                logger.addHandler(file_handler)
            except Exception:
                pass
        return logger

    # Configure basic logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up console handler
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'training.log')
            # Truncate once at initial setup to start fresh, then use append mode
            try:
                with open(log_file, 'w', encoding='utf-8'):
                    pass
            except Exception:
                # If truncation fails, proceed; handler will create/append
                pass
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
        except Exception:
            # Silently continue with console-only logging if file logging fails
            pass
    
    _LOGGING_INITIALIZED = True
    return root_logger
