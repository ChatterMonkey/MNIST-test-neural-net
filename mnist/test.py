import torch
import torch.nn.functional
from mnist.functions import significance_loss,prepare_target,modified_significance_loss

def test(network, data,target,loss_function_id,cutoff = -1,give_sample = False, evaluate_loss = True): #set cutoff to -1 for no cutoff, optional variable
    if loss_function_id == -1:
        evaluate_loss = False
    print("cutoff is {}".format(cutoff))
    network.eval()   #turn off drop off layers etc. for testing

    test_losses = []
    total_number_correct = 0
    sample_output = 0 #this is for finding a good random seed number
    true_positive_count = 0 #number of times that the neural network correctly identified the signal
    false_positive_count = 0 #number of times the neural network misclassified backgroud as signal


    with torch.no_grad():  #turn of gradient decent
        output = network(data) #query the neural network

        if evaluate_loss:
            if loss_function_id ==-1: #this shouldn't happend
                print("ERROR! attempted to run testing with null loss function but neglected to set evaluate_loss=False")
                return "error"

            elif loss_function_id ==0: #mse loss, 2 numbers of full dataset
                loss = torch.nn.functional.mse_loss(torch.reshape(output,(-1,)),prepare_target(target))
            elif loss_function_id ==1: #sig loss
                target_length = torch.unique(target,True,False,False).shape[0]
                if target_length == 2: #only 2 numbers
                    loss = significance_loss(target,output,False) #target preparation happens within significance loss
                elif target_length == 10: #full dataset

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

                    loss = modified_significance_loss(target,output) #target preparation happens within significance loss

                else:
                    print("LESS THEN 10 DIFFERENT SIGNALS APPEARED IN THE TARGET") #suspicious activity
                    return "warning"
            else:
                print("LOSS FUNCTION ID NOT VAID")
                return "LOSS FUNCTION ID NOT VAID"

            test_losses.append(loss.item())

        if give_sample:
            sample_output = output #store a sample output for finding a good seed

        if cutoff == -1: #-1 means use defult of 0.5
            cutoff = 0.5


        for i in range(len(output)):
            if (output[i] > cutoff): # network thinks it is the signal

                output[i] = 1

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
