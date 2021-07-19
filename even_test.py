import torch
import torch.nn.functional
from loaders import test_loader
from variables import signal,test_batch_size,train_batch_size
from functions import significance_loss,prepare_target

def even_test(network, cutoff = -1): #set cutoff to -1 for no cutoff, optional variable
    print("cutoff is {}".format(cutoff))
    network.eval()   #turn off drop off layers etc. for testing
    test_losses = []
    total_number_correct = 0

    sample_output = 0

    true_positive_count = 0 #number of times that the neural network correctly identified the signal
    false_positive_count = 0 #number of times the neural network misclassified backgroud as signal

    with torch.no_grad():  #turn of gradient decent
        for batch, (data, target) in enumerate(test_loader):
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
            #print(prossesed_target_list)

            for i in range(len(prossesed_data_list)):
                prossesed_data[i] = prossesed_data_list[i]
            for i in range(len(prossesed_target_list)):
                prossesed_target[i] = prossesed_target_list[i]


            #print(target)
            #print(prossesed_target)



            #target = prepare_target(target)
            output = network(prossesed_data) #query the neural network

            loss = significance_loss(prossesed_target,output,test_batch_size)
            test_losses.append(loss.item())
            if batch == 0:
                sample_output = output

            if cutoff != -1: # set to -1 for not generating the roc curve
                for i in range(len(output)):

                    if (output[i] > cutoff): # network thinks it is the signal
                        if prossesed_target_list[i] == signal:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal
            else:

                #predicted_values = output.data.max(1, keepdim=True)[1] #pred is a tensor list of all of the neural network's predicted valeus
                for i in range(len(output)):
                    if output[i] > 0.5:
                        output[i] = 1
                    else:
                        output[i] = 0


                #total_number_correct += output.eq(prossesed_target.data.view_as(output)).sum() # .eq compares pred and target and gives true for a match,and false for a miss. .sum() adds num of True, view_as is for the dimentional matc
                for i in range(len(prossesed_target)):
                    if prossesed_target[i] == output[i]:
                        total_number_correct +=1


                for i in range(len(output)):
                    if output[i]==1:
                        if prossesed_target_list[i] == 4:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal

            #loss = significance_loss(prossesed_target,output,test_batch_size)



        return (test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output)
