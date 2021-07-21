import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from net import Net

from loaders import test_loader
import math
from test_mse import test_mse
#for batch, (data, target) in enumerate(train_loader):
#    for batch, (data, target) in enumerate(test_loader):
torch.backends.cudnn.enabled = False

testing_with_full_data = True
loss_function_ids = {"Significance Loss":0,"Mean Squared Error":1}
loss_id = 0

if not testing_with_full_data: #training with only two numbers is only written for sig_loss at the moment
    assert(loss_id !=1)

if testing_with_full_data:
    total_number_of_datapoints = 10000
else:
    total_number_of_datapoints = 2000

def initilize():
    torch.manual_seed(random_seed)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    return network, optimizer


def train_and_test(loss_function_id):
    network, optimizer = initilize()
    test_loss_list = []
    train_loss_list = []

    for epoch in range(1, n_epochs + 1):
        print("Epoch number {}".format(epoch))

        if loss_function_id == 0: #sig loss
            if testing_with_full_data:
                train_losses = train_sigloss(network, optimizer)
            else:
                train_losses = train_even_sigloss(network, optimizer)
        if loss_function_id == 1:
            train_losses = train_mse(network,optimizer)
        else:
            print("LOSS FUNCTION ID NOT RECOGNIZED")

        print("Training Complete, loss: {}".format(train_losses))

        for i in range(len(train_losses)):
            train_loss_list.append(train_losses[i]) #store the loss
            #train_loss_list.append(math.exp(train_losses[i]))

        if loss_function_id == 0:
            if testing_with_full_data:
                test_losses, total_number_correct, true_positive_count, false_positive_count, sample_output = test_sigloss(network, -1) #-1 -> do not generate ROC curve
            else:
                test_losses, total_number_correct, true_positive_count, false_positive_count, sample_output = even_test(network, -1) #-1 -> do not generate ROC curve
                print("test losses {}".format(test_losses))
        #if loss_function_id ==1:
        #    test_losses, total_number_correct, true_positive_count, false_positive_count, sample_output = test_mse(network, -1)

        else:
            print("LOSS FUNCTION ID NOT RECOGNIZED")

        for i in range(len(test_losses)):
            test_loss_list.append(test_losses[i])
            #test_loss_list.append(math.exp(test_losses[i]))

        print("Testing Complete, loss: {}".format(test_losses))

        print("Total number correct: {} percent accuracy: {}, true positives {}, false positives {}".format(total_number_correct,100 * total_number_correct / total_number_of_datapoints,true_positive_count,false_positive_count))
        torch.save(network.state_dict(), '/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/neuralnets/model2.pth')


    plt.subplot(1, 2, 1)
    plt.xlabel("Batch # (15 batches per epoch)")
    plt.ylabel("Training Loss")
    # plt.ylim(-10,10)
    plt.plot(train_loss_list)
    #plt.savefig("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/matplotlib_output/lr0.001train.png")


    plt.subplot(1, 2, 2)
    plt.xlabel("Batch # (10 batches per epoch)")
    plt.ylabel("Test Loss")
    plt.plot(test_loss_list)
    print(test_loss_list)
    plt.savefig("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/matplotlib_output/loss.png")


train_and_test(0)
