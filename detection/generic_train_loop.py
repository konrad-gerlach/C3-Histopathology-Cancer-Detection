import torch

def training_loss_function(outputs,y):
    return nn.BCEWithLogitsLoss()(outputs[-1],y)

def train_loop(X, y, device, model, logger, metrics, gradient_accumulation=1, optimizer=None, batch=0):

    loss_fn = training_loss_function

    # Compute prediction and loss
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    y = y.view(-1, 1).to(torch.float)

    outputs = model.forward_per_layer(X)
    loss = loss_fn(outputs, y)

    # Backpropagation with gradient accumulation
    if optimizer:
        loss.backward()
        if batch % gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    metrics = logger(outputs, loss, batch, X, y, metrics)
    return metrics
