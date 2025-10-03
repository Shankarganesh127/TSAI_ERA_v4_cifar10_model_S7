
# train_test.py
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging


class train_test_model:
    
    def __init__(self, model, device, train_loader, test_loader,criterion,optimizer,scheduler,epochs=1):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_acc_list = []
        self.test_acc_list = []
        self.F = F  # Assign torch.nn.functional to self.F for easier access
        self.epochs = epochs

    def train(self, model, device, train_loader, optimizer, criterion,epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc="Training", leave=True)
        train_loss, correct, processed = 0, 0, 0
        # Train with progress bar
        
        for batch_idx, (data, target) in enumerate(pbar, 1):
            # get samples and move to device
            data, target = data.to(self.device), target.to(self.device)
            # Initialize optimizer
            self.optimizer.zero_grad()
            # Prediction
            output = self.model(data)
            # Calculate loss
            loss = self.F.nll_loss(output, target)
            # Backpropagation
            loss.backward()
            # Gradient clipping to stabilize higher LR OneCycle swings
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            # Per-batch scheduler stepping (e.g., OneCycleLR) if attribute present
            if hasattr(self, 'scheduler') and getattr(self.scheduler, 'batch_step', False):
                self.scheduler.step()
            # -----------------------------
            # Accumulate loss and calculate accuracy
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
            processed += len(data)
            
            # Calculate current metrics
            current_loss = train_loss / processed
            current_accuracy = 100. * correct / processed
            
            # Update progress bar only (add LR peek occasionally)
            if hasattr(self, 'scheduler') and getattr(self.scheduler, 'batch_step', False):
                current_lr = self.scheduler.get_last_lr()[0]
                status = f"Train Loss={current_loss:.4f} Acc={current_accuracy:.2f}% LR={current_lr:.4f}"
            else:
                status = f"Train Loss={current_loss:.4f} Accuracy={current_accuracy:.2f}%"
            pbar.set_description(desc=status)

        # Final epoch-level logging for training metrics
        epoch_accuracy = 100. * correct / len(self.train_loader.dataset)
        logging.info(
            f'Epoch {epoch:02d}/{self.epochs}: Train set final results: Average loss: {train_loss:.4f}, '
            f'Accuracy: {correct}/{len(self.train_loader.dataset)} ({epoch_accuracy:.2f}%)'
        )
        return epoch_accuracy

    def test(self, model, device, test_loader, criterion,epoch):
        self.model.eval()
        test_loss, correct = 0, 0
        # Test with progress bar
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing", leave=True)
            for batch_idx, (data, target) in enumerate(pbar, 1):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()
                correct += batch_correct
                
                # Calculate current metrics
                current_loss = test_loss / (batch_idx * len(data))
                current_accuracy = 100. * correct / (batch_idx * len(data))
                
                # Update progress bar only
                status = f"Test Loss={current_loss:.4f} Accuracy={current_accuracy:.2f}%"
                pbar.set_description(desc=status)
                
        test_loss /= len(self.test_loader.dataset)
        acc = 100. * correct / len(self.test_loader.dataset)
        logging.info(f'Epoch {epoch:02d}/{self.epochs}:Test set final results: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({acc:.2f}%)')
        return acc

    def do_training(self,epoch):
        return self.train(self.model, self.device, self.train_loader, self.optimizer, self.criterion,epoch)

    def do_testing(self,epoch):
        return self.test(self.model, self.device, self.test_loader, self.criterion,epoch)

    def run_epoch(self):
        logging.info(f"Training model for {self.epochs} epochs")
        for epoch in range(1, self.epochs+1):
            train_acc = self.do_training(epoch=epoch)
            test_acc = self.do_testing(epoch=epoch)
            # Epoch-level scheduler step only if not using per-batch scheduler
            if hasattr(self.scheduler, 'batch_step') and not getattr(self.scheduler, 'batch_step'):
                self.scheduler.step()
            elif not hasattr(self.scheduler, 'batch_step'):
                # Legacy schedulers
                try:
                    self.scheduler.step()
                except Exception:
                    pass
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            
            #logging.info(f"Epoch {epoch:02d}/{self.epochs}: Train={train_acc:.2f}%, Test={test_acc:.2f}%, LR={self.scheduler.get_last_lr()[0]:.6f}")
            
    def plot_results(self):
        plt.plot(self.train_acc_list, label='Train Acc')
        plt.plot(self.test_acc_list, label='Test Acc')
        plt.legend()
        plt.title("Training vs Test Accuracy")
        plt.show()



