import torch
import helper
import wandb

def train_loop(batch,X,y, device, model,loss_fn, gradient_accumulation,optimizer,logger,inputs):

    # Compute prediction and loss
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    y = y.view(-1, 1).to(torch.float)

    outputs = model.forward_per_layer(X)
    loss = loss_fn(outputs, y)

    # Backpropagation with gradient accumulation
    loss.backward()
    if batch % gradient_accumulation == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    inputs = logger(outputs,loss,batch,X,y,inputs)
    return inputs
