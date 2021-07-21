import torch
import torch.nn.functional
from loaders import test_loader
from variables import signal,test_batch_size,train_batch_size
from functions import significance_loss,prepare_target

def test_mse(network, cutoff = -1): #set cutoff to -1 for no cutoff, optional variable
    print("cutoff is {}".format(cutoff))
    network.eval()   #turn off drop off layers etc. for testing
    test_losses = []
    total_number_correct = 0
    sample_output = 0 #this is for finding a good random seed number
    true_positive_count = 0 #number of times that the neural network correctly identified the signal
    false_positive_count = 0 #number of times the neural network misclassified backgroud as signal

    with torch.no_grad():  #turn of gradient decent
        for batch, (data, target) in enumerate(test_loader):
            prossesed_data_list = [] # a list of the images that are either 4 or 7
            prossesed_target_list = [] # 4 and 7's from the target

            for i in range(len(target)):
                if target[i] == 4:
                    prossesed_target_list.append(4)
                    prossesed_data_list.append(data[i])
                if target[i] == 7:
                    prossesed_target_list.append(7)
                    prossesed_data_list.append(data[i])
            prossesed_data = torch.zeros(len(prossesed_data_list),1,28,28) #create a tensor of the subset of data that is 4 or 7
            prossesed_target = torch.zeros(len(prossesed_target_list)) #tensor subset of the targets

            for i in range(len(prossesed_data_list)):
                prossesed_data[i] = prossesed_data_list[i]
            for i in range(len(prossesed_target_list)):
                prossesed_target[i] = prossesed_target_list[i]
            for i in range(10):
                print(prossesed_target[i])



            output = network(prossesed_data) #query the neural network
            loss = torch.nn.functional.mse_loss(torch.reshape(output,(-1,)),prossesed_target)
            test_losses.append(loss.item())

            if batch == 0:
                sample_output = output #store a sample output for finding a good seed

            if cutoff != -1: # set to -1 for not generating the roc curve
                for i in range(len(output)):
                    if (output[i] > cutoff): # network thinks it is the signal
                        if target[i] == signal:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal
            else:

                for i in range(len(output)): # convert the output into a binery classification 1 for signal and 0 for background
                    if output[i] > 0.5:
                        output[i] = 1
                    else:
                        output[i] = 0


                #total_number_correct += output.eq(prossesed_target.data.view_as(output)).sum() # .eq compares pred and target and gives true for a match,and false for a miss. .sum() adds num of True, view_as is for the dimentional matc
                target = prepare_target(target) # convert target to 1's and 0's
                for i in range(len(target)): #how acurate was the nn overall?
                    if target[i] == output[i]:
                        total_number_correct += 1

                for i in range(len(output)): #find false positives verses true positives
                    if output[i]==1:
                        if target[i] == 1:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal


        return (test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output)
