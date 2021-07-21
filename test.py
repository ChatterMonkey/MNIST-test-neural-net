import torch
import torch.nn.functional
from variables import signal
from functions import significance_loss,prepare_target

def test(network, data,target,loss_function_id,cutoff = -1,give_sample = False): #set cutoff to -1 for no cutoff, optional variable
    print("cutoff is {}".format(cutoff))
    network.eval()   #turn off drop off layers etc. for testing
    test_losses = []
    total_number_correct = 0
    sample_output = 0 #this is for finding a good random seed number
    true_positive_count = 0 #number of times that the neural network correctly identified the signal
    false_positive_count = 0 #number of times the neural network misclassified backgroud as signal
    print(target)

    with torch.no_grad():  #turn of gradient decent
        output = network(data) #query the neural network

        if loss_function_id ==0: #mse loss, 2 numbers of full dataset
            loss = torch.nn.functional.mse_loss(torch.reshape(output,(-1,)),prepare_target(target))
        elif loss_function_id ==1: #sig loss
            target_length = torch.unique(target,True,False,False).shape[0]
            if target_length == 2: #only 2 numbers
                loss = significance_loss(target,output,False) #target preparation happens within significance loss
            elif target_length == 10: #full dataset
                print(target)
                print(output)
                loss = significance_loss(target,output,True) #target preparation happens within significance loss
            else:
                print("LESS THEN 10 DIFFERENT SIGNALS APPEARED IN THE TARGET") #suspicious activity
                return "warning"
        elif loss_function_id == 2: #binery cross entropy
            loss = torch.nn.functional.binary_cross_entropy(torch.reshape(output,(-1,)),prepare_target(target))
        else:
            print("LOSS FUNCTION ID NOT VAID")
            return "LOSS FUNCTION ID NOT VAID"


        print(target)

        test_losses.append(loss.item())

        if give_sample:
            sample_output = output #store a sample output for finding a good seed

        if cutoff == -1: #-1 means use defult of 0.5
            cutoff = 0.5
            #print("default otpion")

        for i in range(len(output)):
            if (output[i] > cutoff): # network thinks it is the signal
                #print(output[i])
                output[i] = 1
                #print(output[i])
                #print(target[i])
                if target[i] == 1:
                    true_positive_count += 1 #neural network correctly classified the signal
                    total_number_correct += 1 #network correctly identified signal
                else:
                    false_positive_count += 1 #neural network thought the background was signal
            else:
                output[i] = 0
                if target[i] ==0:
                    total_number_correct += 1 #network correctly identified background

        if give_sample:
            return (test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output)
        else:
            return (test_losses,total_number_correct,true_positive_count,false_positive_count)
