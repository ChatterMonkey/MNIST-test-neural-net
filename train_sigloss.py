import torch
from loaders import train_loader
from variables import *
from loss_functions import significance_loss



def train_sigloss(network, optimizer): #train with the significance loss

    network.train() #turn on training specific stuff like dropout layers
    loss_function = significance_loss(train_batch_size/10,9*train_batch_size/10)
    loss_for_each_batch = []

    for batch, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = network(data)
        target_list = torch.zeros(len(target)) # list with 1 for signal and 0 for background
        prediction_list = torch.zeros(len(output)) #list of predictions with 1 for signal and 0 for background

        for i in range(len(target)): #setup the target list
            if target[i] == signal:
                target_list[i] = 1
        for i in range(len(output)): #setup the prediction list
            if output[i].argmax().item() == signal:
                prediction_list[i] = 1

        target_list.requires_grad_(True)
        prediction_list.requires_grad_(True) #nessecary to compute derivatives for backpropagation

        loss = loss_function(target_list,prediction_list)

        if batch % log_interval == 0:
            print("Training batch {}/{}, loss was {}".format(batch,60000/train_batch_size,loss))
            #torch.save(network.state_dict(), '/Users/mayabasu/results/model.pth')
            #torch.save(optimizer.state_dict(), '/Users/mayabasu/results/optimizer.pth')

        loss_for_each_batch.append(loss)
        loss.backward()
        optimizer.step()

    return loss_for_each_batch




