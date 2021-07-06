import torch
import torch.nn.functional
from loaders import test_loader
from variables import signal

def test(network, num_test_batches, test_batch_size):
    print("testing...")
    network.eval()   #turn off drop off layers etc. for testing
    test_loss = 0
    correct = 0
    total = 0
    test_losses =[]

    num_times_signal_appears_in_dataset = 0
    num_times_signal_was_missed = 0
    num_times_signal_was_contaminated = 0

    with torch.no_grad():  #turn of gradient decent
        for batch, (data, target) in enumerate(test_loader):
            print("batch {} number of test batches {} are they equal {}",batch, num_test_batches, batch ==num_test_batches)

            if batch < num_test_batches:
                print("Test batch {}".format(batch))
                output = network(data) #query the neural network
                test_loss += torch.nn.functional.nll_loss(output, target, size_average=False).item() #calculate loss
                test_loss /= test_batch_size
                test_losses.append(test_loss)

                pred = output.data.max(1, keepdim=True)[1] #pred is a tensor list of all of the neural network's predicted valeus
                correct += pred.eq(target.data.view_as(pred)).sum() # .eq compares pred and target and gives true for a match
                total += 1

                                                                    # and false for a miss. .sum() adds num of True, view_as is for the dimentional match

                for i in range(len(target)):
                    if target[i] == signal:
                        num_times_signal_appears_in_dataset +=1

                        if target[i] != pred[i]: # if the value was signal but it was missed
                            num_times_signal_was_missed += 1
                    else: #other number is presented to the network
                        if pred[i] == signal: #nn thinks it is a signal
                            num_times_signal_was_contaminated += 1

            else:
                print("last Test batch {}".format(batch))
                print("The neural network got {} out of {} correct in this batch, {}% correct".format(correct, total, correct / test_batch_size * 100))
                print("Testing loss: {}".format(test_losses))
                print("The signal was contaminated {} times and the signal was missed {} times".format(num_times_signal_was_contaminated,num_times_signal_was_missed))

                return (correct, test_losses,num_times_signal_was_missed,num_times_signal_was_contaminated,num_times_signal_appears_in_dataset)
        print("all batches used")
        print("last Test batch {}".format(batch))
        print("The neural network got {} out of {} correct in this batch, {}% correct".format(correct, total, correct / test_batch_size * 100))
        print("Testing loss: {}".format(test_losses))
        print("The signal was contaminated {} times and the signal was missed {} times".format(num_times_signal_was_contaminated,num_times_signal_was_missed))

        return (correct, test_losses,num_times_signal_was_missed,num_times_signal_was_contaminated,num_times_signal_appears_in_dataset)

