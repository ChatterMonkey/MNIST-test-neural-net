import torch
import torch.nn.functional
from functions import prepare_target
from loaders import train_loader
from variables import *

def train_mse(network, optimizer):
    network.train()

    train_losses = []


    for batch, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        target = prepare_target(target) #convert target into 1 for siganl and 0 for background
        output = torch.reshape(network(data), (-1,)) #query network and reshape output
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()


        if batch % 10 == 0:
            train_losses.append(loss.item())


        optimizer.step()

        if batch % log_interval == 0:
            print("Training batch {}/{}, Loss: {}".format(batch,60000/train_batch_size,loss))
            #torch.save(network.state_dict(), '/Users/mayabasu/results/model.pth')
            #torch.save(optimizer.state_dict(), '/Users/mayabasu/results/optimizer.pth')

    return train_losses
