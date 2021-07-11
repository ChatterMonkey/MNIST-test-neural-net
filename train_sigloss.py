import torch
from torch.nn import functional as F


from loaders import train_loader
from variables import *
from loss_functions import significance_loss


def train(network, optimizer, num_train_batches, train_batch_size, epoch):

    network.train()
    train_losses = []

    for batch, (data, target) in enumerate(train_loader):

        #if (batch < num_train_batches):
        optimizer.zero_grad()
        output = network(data)

        training_weight_list = []

        for i in range(10):
            if i == signal:
                training_weight_list.append(signal_weight_strength)
                #training_weight_list.append(1)
            else:
                #training_weight_list.append((1-signal_weight_strength)/9)
                training_weight_list.append(1)

        training_weight = torch.tensor(training_weight_list).float()
        loss_target = torch.tensor()
        for i in range(len(target)):
            if target[i] == signal:
                loss_target.add(1)
            else:
                loss_target.add(0)

        loss_pred = torch.tensor()
        for i in range(len(output)):
            if output[i].argmax().item ==signal:
                loss_pred.add(1)
            else:
                loss_pred.add(0) #c




        #loss = F.nll_loss(output, target,weight=training_weight)
        #loss = F.nll_loss(output, target)
        loss_function = significance_loss(train_batch_size/10,9*train_batch_size/10)
        loss = loss_function(loss_target,loss_pred)
        loss.backward()
        optimizer.step()

        if batch % log_interval ==0:
            print("Training batch {}".format(batch))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * train_batch_size, len(train_loader.dataset), 100. * batch / len(train_loader), loss.item()))
        train_losses.append(loss.item())

        torch.save(network.state_dict(), '/Users/mayabasu/results/model.pth')
        torch.save(optimizer.state_dict(), '/Users/mayabasu/results/optimizer.pth')
       # else:
        #    print("break")
          #  break

    return train_losses
