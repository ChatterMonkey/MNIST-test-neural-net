import torch
import torch.nn.functional
from loaders import test_loader
from variables import signal,test_batch_size,train_batch_size
from functions import significance_loss,prepare_target

def test(network, cutoff = -1): #set cutoff to -1 for no cutoff, optional variable
    print("cutoff is {}".format(cutoff))
    network.eval()   #turn off drop off layers etc. for testing
    test_losses = []
    total_number_correct = 0

    sample_output = 0

    true_positive_count = 0 #number of times that the neural network correctly identified the signal
    false_positive_count = 0 #number of times the neural network misclassified backgroud as signal

    with torch.no_grad():  #turn of gradient decent
        for batch, (data, target) in enumerate(test_loader):
            thinks_are_signal = 0
            #target = prepare_target(target)
            output = network(data) #query the neural network
            if batch == 0:
                sample_output = output

            if cutoff != -1: # set to -1 for not generating the roc curve
                for i in range(len(output)):

                    if (output[i] > cutoff): # network thinks it is the signal
                        if target[i] == signal:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal
            else:

                #predicted_values = output.data.max(1, keepdim=True)[1] #pred is a tensor list of all of the neural network's predicted valeus
                for i in range(len(output)):
                    if output[i] > 0.5:
                        thinks_are_signal += 1
                        output[i] = 1
                    else:
                        output[i] = 0
                print("THINKS ARE SIGNAL {}".format(thinks_are_signal))

                total_number_correct += output.eq(target.data.view_as(output)).sum() # .eq compares pred and target and gives true for a match,and false for a miss. .sum() adds num of True, view_as is for the dimentional matc
                print("total number correct")
                print(total_number_correct)
                for i in range(len(output)):
                    if output[i]==1:
                        if target[i] == 1:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal

            loss = significance_loss(target,output,test_batch_size)
            test_losses.append(loss.item())
            print("LOSSS {}".format(loss.item()))

        return (test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output)
