import torch
from loaders import train_loader
from variables import *
from functions import significance_loss,prepare_target
import torch.nn.functional

def train_sigloss(network, optimizer):  # train with the significance loss
    print("TRAINING WITH SIGLOSS")

    network.train()  # turn on training specific stuff like dropout layers
    loss_for_each_batch = []

    for batch, (data, target) in enumerate(train_loader):
        print("data")
        print(target.size())
        prossesed_data_list = []
        prossesed_target_list = []

        for i in range(len(target)):
            if target[i] == 4:
                prossesed_target_list.append(4)
                prossesed_data_list.append(data[i])
            if target[i] ==7:
                prossesed_target_list.append(7)
                prossesed_data_list.append(data[i])
        prossesed_data = torch.zeros(len(prossesed_data_list),1,28,28)
        prossesed_target = torch.zeros(len(prossesed_target_list))
        print(prossesed_target_list)

        for i in range(len(prossesed_data_list)):
            prossesed_data[i] = prossesed_data_list[i]
        for i in range(len(prossesed_target_list)):
            prossesed_target[i] = prossesed_target_list[i]


        print(target)
        print(prossesed_target)





        optimizer.zero_grad()
        output = network(prossesed_data)

        #loss = significance_loss(prossesed_target, output, train_batch_size)
        loss = torch.nn.functional.mse_loss(torch.reshape(output,(-1,)), prepare_target(prossesed_target))

        if batch % log_interval == 0:
            print("Training batch {}/{}, loss was {}".format(batch, 60000 / train_batch_size, loss))
            # torch.save(network.state_dict(), '/Users/mayabasu/results/model.pth')
            # torch.save(optimizer.state_dict(), '/Users/mayabasu/results/optimizer.pth')
        loss.backward()
        print("grad of output")

        optimizer.step()

        loss_for_each_batch.append(loss.item())

    return loss_for_each_batch
