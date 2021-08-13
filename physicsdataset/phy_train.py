import torch.nn.functional as f
from physicsdataset.phy_functions import sig_loss
import torch

def train(network, optimizer, data, target, loss_function_id):
    network.train()
    optimizer.zero_grad()
    output = network(data)
    if loss_function_id == 0:
        loss = f.mse_loss(output, target)
    elif loss_function_id == 1:
        loss = sig_loss(output,target)
    elif loss_function_id == 2:
        loss = f.binary_cross_entropy(output,target)
    else:
        print("LOSS FUNCTION ID NOT VAID")
        return "LOSS FUNCTION ID NOT VAID"

    loss.backward()
    optimizer.step()
    return loss.item()
