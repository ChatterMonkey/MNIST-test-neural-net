import torch
from loaders import train_loader
from variables import *
from functions import significance_loss



def train_sigloss(network, optimizer): #train with the significance loss

    network.train() #turn on training specific stuff like dropout layers
    loss_function = significance_loss(train_batch_size/10,9*train_batch_size/10)
    loss_for_each_batch = []

    for batch, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = network(data)

        loss = significance_loss(target,output,train_batch_size)

        if batch % log_interval == 0:
            print("Training batch {}/{}, loss was {}".format(batch,60000/train_batch_size,loss))
            #torch.save(network.state_dict(), '/Users/mayabasu/results/model.pth')
            #torch.save(optimizer.state_dict(), '/Users/mayabasu/results/optimizer.pth')

        loss_for_each_batch.append(loss)
        loss.backward()
        optimizer.step()

    return loss_for_each_batch




