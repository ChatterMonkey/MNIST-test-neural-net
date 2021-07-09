import torch
import torch.nn.functional
from loaders import test_loader
from variables import signal

def test(network, num_test_batches, test_batch_size,cutoff): #set cutoff to 0 for no cutoff
    print("testing...")
    network.eval()   #turn off drop off layers etc. for testing
    test_loss = 0
    correct = 0
    total = 0
    test_losses =[]
    set_of_deviations = []

    num_times_signal_appears_in_dataset = 0
    num_times_signal_was_missed = 0
    num_times_signal_was_contaminated = 0
    true_positive_count = 0
    false_positive_count = 0
    hits = 0

    with torch.no_grad():  #turn of gradient decent
        for batch, (data, target) in enumerate(test_loader):
            print("batch {} number of test batches {} are they equal {}".format(batch, num_test_batches, batch ==num_test_batches))

            if batch < num_test_batches:
                print("Test batch {}".format(batch))

                output = network(data) #query the neural network

                #print(output[1])
                #print(torch.sigmoid(output[1]))
                #print(output[1].max())
                #print(target[1])


                if cutoff != -1:
                    for i in range(len(output)):
                        soft_output = torch.nn.Softmax(dim=0)(output[i])
                        #print("soft output {}".format(soft_output))

                        if (soft_output[signal] > cutoff):



                            if target[i] == signal:
                                true_positive_count += 1
                            else:
                                false_positive_count +=1
                        #else:
                            #print("not large enough")
                            #print("cut off was {} output was {}".format(cutoff,torch.nn.Softmax(dim=0)(output[i][signal])))




                test_loss += torch.nn.functional.nll_loss(output, target, size_average=False).item() #calculate loss
                test_loss /= test_batch_size
                test_losses.append(test_loss)

                pred = output.data.max(1, keepdim=True)[1] #pred is a tensor list of all of the neural network's predicted valeus
                correct += pred.eq(target.data.view_as(pred)).sum() # .eq compares pred and target and gives true for a match
                total += 1

                                                                    # and false for a miss. .sum() adds num of True, view_as is for the dimentional match

                for i in range(len(target)):
                    if target[i] == signal:
                        num_times_signal_appears_in_dataset += 1
                        #print(target[i])
                        #print(pred[i])
                        if pred[i] ==signal:
                            hits += 1
                            #print(hits)


                        if target[i] != pred[i]: # if the value was signal but it was missed
                            num_times_signal_was_missed += 1 #false negative
                    else: #other number is presented to the network
                        if pred[i] == signal: #nn thinks it is a signal
                            num_times_signal_was_contaminated += 1 #false positive

            else:
                print("last Test batch {}".format(batch))
                print("The neural network got {} out of {} correct in this batch, {}% correct".format(correct, total, correct / test_batch_size * 10))
                print("Testing loss: {}".format(test_losses))
                print("The signal was contaminated {} times and the signal was missed {} times out of {}".format(num_times_signal_was_contaminated,num_times_signal_was_missed,total))

                return (hits,correct, test_losses,num_times_signal_was_missed,num_times_signal_was_contaminated,num_times_signal_appears_in_dataset, false_positive_count,true_positive_count)
        print("all batches used")
        print("last Test batch {}".format(batch))
        print("The neural network got {} out of {} correct in this batch, {}% correct".format(correct, total, correct / test_batch_size * 10))
        print("Testing loss: {}".format(test_losses))
        print("The signal was contaminated {} times and the signal was missed {} times".format(num_times_signal_was_contaminated,num_times_signal_was_missed))

        return (hits,correct, test_losses,num_times_signal_was_missed,num_times_signal_was_contaminated,num_times_signal_appears_in_dataset, false_positive_count,true_positive_count)

