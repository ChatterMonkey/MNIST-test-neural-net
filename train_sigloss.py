import torch
from loaders import train_loader
from variables import *
from functions import significance_loss
import torch.nn.functional

def train_sigloss(network, optimizer):  # train with the significance loss
    print("Training with significance loss")

    network.train()  # turn on training specific stuff like dropout layers
    loss_for_each_batch = []

    for batch, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = network(data)

        loss = significance_loss(target,output,train_batch_size)
        loss_for_each_batch.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch % log_interval == 0:
            print("Training batch {}/{}, loss was {}".format(batch, 60000 / train_batch_size, loss))
            # torch.save(network.state_dict(), '/Users/mayabasu/results/model.pth')
            # torch.save(optimizer.state_dict(), '/Users/mayabasu/results/optimizer.pth')

    return loss_for_each_batch
