### Notes

* Make sure the gradients are cleaned in the training loop with ```optimizer.zero_grad()```;
* In validation loop, the network must be in the evaluation mode with ```model.eval()```, then back to training mode with ```model.train()```;
* PyTorch can only perform operations on tensors that are on the same device, so either both CPU or both GPU;