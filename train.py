import torch
from loaders import train_loader
from variables import *
import torch.nn.functional


def train(network, optimizer):
    network.train()
    train_losses = []

    for batch, (data, target) in enumerate(train_loader):
        for i in range(len(target)):
            if target[i] == signal:
                target[i] = 1

            else:
                target[i] = 0

        optimizer.zero_grad()
        output = network(data)
        # print(output)

        output = output.float()
        target = target.float()
        output = torch.reshape(output, (-1,))

        print("TARGET")
        print(target)

        loss = torch.nn.functional.mse_loss(output, target)
        if batch%10 ==0:
            train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch % log_interval == 0:
            print("LOSS IS {}".format(loss))
            print("Training batch {}".format(batch))

            torch.save(network.state_dict(), '/Users/mayabasu/results/model.pth')
            torch.save(optimizer.state_dict(), '/Users/mayabasu/results/optimizer.pth')

    return train_losses
