import torch
import torch.nn.functional
from functions import significance_loss,prepare_target
from loaders import train_loader
from variables import *

def train(network, optimizer, data,target, loss_function_id):

    network.train()
    optimizer.zero_grad()
    output = torch.reshape(network(data), (-1,)) #query network and reshape output

    if loss_function_id == 0:
        target = prepare_target(target) #convert target into 1 for siganl and 0 for background
        loss = torch.nn.functional.mse_loss(output, target)
    if loss_function_id == 1:
        if len(set(target)) == 2: #only 2 numbers
            target = prepare_target(target)
            loss = significance_loss(target,output,False) #target preparation happens within significance loss
        if len(set(target)) ==10: #full dataset
            target = prepare_target(target)
            loss = significance_loss(target,output,True) #target preparation happens within significance loss
        else:
            print("LESS THEN 10 DIFFERENT SIGNALS APPEARED IN THE TARGET") #suspicious activity
            return "warning"

    loss.backward()
    optimizer.step()

    return loss.item()
