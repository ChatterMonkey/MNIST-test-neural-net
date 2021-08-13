import torch
import torch.nn.functional
from mnist.functions import significance_loss,prepare_target,modified_significance_loss


def train(network, optimizer, data,target, loss_function_id):

    network.train()
    optimizer.zero_grad()
    output = torch.reshape(network(data), (-1,)) #query network and reshape output

    if loss_function_id == 0:
        target = prepare_target(target) #convert target into 1 for siganl and 0 for background
        loss = torch.nn.functional.mse_loss(output, target)
    elif loss_function_id == 1:
        #print(target)
        target_length = torch.unique(target,True,False,False).shape[0]
        #print(target_length)
        if target_length == 2: #only 2 numbers

            loss = significance_loss(target,output,False) #target preparation happens within significance loss
        elif target_length ==10: #full dataset

            loss = significance_loss(target,output,True) #target preparation happens within significance loss

        else:
            print("LESS THEN 10 DIFFERENT SIGNALS APPEARED IN THE TARGET") #suspicious activity
            return "warning"
    elif loss_function_id == 2: #binery cross entropy
        loss = torch.nn.functional.binary_cross_entropy(torch.reshape(output,(-1,)),prepare_target(target))
    elif loss_function_id ==3:
        target_length = torch.unique(target,True,False,False).shape[0]
        #print(target_length)
        if target_length == 2: #only 2 numbers

            loss = modified_significance_loss(target,output,False) #target preparation happens within significance loss
        elif target_length ==10: #full dataset

            loss = modified_significance_loss(target,output,True) #target preparation happens within significance loss

        else:
            print("LESS THEN 10 DIFFERENT SIGNALS APPEARED IN THE TARGET") #suspicious activity
            return "warning"
    else:
        print("LOSS FUNCTION ID NOT VAID")
        return "LOSS FUNCTION ID NOT VAID"

    loss.backward()
    optimizer.step()

    return loss.item()
