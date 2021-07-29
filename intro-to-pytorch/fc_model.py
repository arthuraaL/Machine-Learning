import torch
from torch import nn, optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])

        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x, is_flattened=False):
        if not is_flattened:
            # Ensuring that the input is flattened
            x = x.view(x.shape[0], -1)
        
        for hidden_layer in self.hidden_layers:
            x = self.dropout(F.relu(hidden_layer(x)))
        x = F.log_softmax(self.output(x), dim=1)
        return x

def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        log_ps = model(images)
        test_loss += criterion(log_ps, labels).item()

        ## Calculating the accuracy 
        # Output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(log_ps)
        # topk will return the most likely k classes (in this case 1)
        top_values, top_indices = ps.topk(1, dim=1)
        # Class with highest probability is our predicted class, compare with true label
        equals = top_indices == labels.view(-1, 1)
        # equals has type torch.ByteTensor but torch.mean isn't implemented for tensors with that type
        equals = equals.float()
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += torch.mean(equals)

    return test_loss, accuracy

def train(model, trainloader, testloader, criterion, optimizer, epochs=5):
    for e in range(epochs):
        # In training mode, Dropout is on
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            # Zero out the gradients to prevent them from accumulating
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Turn off the gradients to speed up
        with torch.no_grad():
            # In inference mode, Dropout is off
            model.eval()
            test_loss, accuracy = validation(model, testloader, criterion)
            print(f'Epoch: {e+1}/{epochs} --- ',
                  f'Training Loss: {running_loss/len(trainloader)} --- ',
                  f'Validation Loss: {test_loss/len(testloader)} --- ',
                  f'Validation Accuracy: {accuracy.item()/len(testloader)*100}%')
        
        # Turn back to the training mode (Dropount and gradients are back to the game)
        model.train()

