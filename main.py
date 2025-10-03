
# main.py
import logging
from logger_setup import setup_logging
import summarizer
import io
import contextlib
import train_test
import cifar10model_v0


class get_model:
    def __init__(self,device=None):
        self.device = device if device else self.get_device()
        self.model_obj = self.get_model()
        self.model_config = self.get_config()

    def get_device(self):
        return train_test.torch.device("cuda" if train_test.torch.cuda.is_available() else "cpu")

    def get_model(self):
        return cifar10model_v0.Net().to(self.device)

    def get_config(self):
        return cifar10model_v0.set_config_v0().setup(self.model_obj)

def main_i(params_check=1):
    logging.info("Setting up for model")
    model = get_model(device=None)
    # Capture printed summary into logs
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
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
    params_check = int(input("Enter 1 for params check only, 0 for full training/testing: "))
    main_i(params_check=params_check)

if __name__ == "__main__":
    main()
