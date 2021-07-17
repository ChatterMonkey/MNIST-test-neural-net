import torch
import torch.nn.functional
from loaders import test_loader
from variables import signal,test_batch_size,train_batch_size
from functions import significance_loss

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

            output = network(data) #query the neural network
            if batch == 0:
                sample_output = output




            if cutoff != -1: # set to -1 for not generating the roc curve
                for i in range(len(output)):
                    soft_output = torch.nn.Softmax(dim=0)(output[i])
                    if (soft_output[signal] > cutoff):
                        if target[i] == signal:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal
            else:

                predicted_values = output.data.max(1)[1] #pred is a tensor list of all of the neural network's predicted valeus
                #print(output)
                #print(predicted_values)
                #print(target)
                total_number_correct += predicted_values.eq(target.data.view_as(predicted_values)).sum() # .eq compares pred and target and gives true for a match,and false for a miss. .sum() adds num of True, view_as is for the dimentional matc
                print(total_number_correct)
                for i in range(len(target)):
                    if predicted_values[i]==signal:
                        if target[i] == signal:
                            true_positive_count += 1 #neural network correctly classified the signal
                        else:
                            false_positive_count += 1 #neural network thought the background was signal

            loss = significance_loss(target,output,test_batch_size)
            test_losses.append(loss.item())
            print("LOSSS {}".format(loss.item()))

        return (test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output)
