class Solver():
    def __init__(self, model, loader, optimizer, criterion, **kwargs):
        self.model = model
        self.train_loader = loader['train_loader']
        self.val_loader = loader['val_loader']
        self.optimizer = optimizer
        self.criterion = criterion
        # Unpack keyword arguments
        self.optim_config = kwargs.pop("optim_config", {})
        self.epochs = kwargs.pop("epochs", 20)
        self.verbose = kwargs.pop("verbose", True)
        
        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError(f'Unrecognized arguments {extra}')
            
    def _save_checkpoint(self):
        checkpoint = {
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss_history': self.loss_history,
                      'train_acc_history': self.train_acc_history,
                      'val_acc_history': self.val_acc_history
                      }
        
        filename = f'model.pth'
        torch.save(checkpoint, filename)
        if self.verbose:
            print(f'Saving checkpoint to {filename}')
            
    def _check_accuracy(self, y_pred, y_truedef save_model(self):
        assert y_pred.shape == y_true.shape
        y_pred_rounded = torch.round(torch.sigmoid(y_pred))
        correct_values = (y_pred_rounded == y_true).sum().float()
        acc = correct_values / len(y_true)
        return acc
    
    def train(self):
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        for e in range(1, self.epochs):
            running_loss = 0
            train_acc = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero out the parameters
                self.optimizer.zero_grad()
                # Forward
                outputs = self.model(inputs.float())
                # Calculating the training accuracy
                train_acc += self._check_accuracy(outputs, labels.float())
                # Backward + Optimization
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()
            
                running_loss += loss.item()
            self.loss_history.append(running_loss/len(self.train_loader))
            self.train_acc_history.append(train_acc/len(self.train_loader))
            # Evaluating on the validation set
            val_acc = 0
            with torch.no_grad():
                model.eval()
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward
                    outputs = self.model(inputs.float())
                    # Calculating the validation accuracy
                    val_acc += self._check_accuracy(outputs, labels.float())
                    
                self.val_acc_history.append(val_acc/len(self.val_loader))
                if self.best_val_acc < (val_acc / len(self.val_loader)):
                    self.best_val_acc = val_acc / len(self.val_loader)
                    self._save_checkpoint()
            
            model.train()
            
            if self.verbose:
                print(f'Epoch: {e}/{self.epochs} | Train Acc: {self.train_acc_history[-1]:.2f} | Val Acc: {self.val_acc_history[-1]:.2f}')            