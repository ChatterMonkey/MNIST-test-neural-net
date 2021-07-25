import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from net import Net
from variables import *
from train import train
from test import test
from loaders import test_loader,train_loader
from functions import subset_data
import math
import os
#for batch, (data, target) in enumerate(train_loader):
#    for batch, (data, target) in enumerate(test_loader):
torch.backends.cudnn.enabled = False

using_full_data = True
use_auto_stop = True # automaticallly stop when accuracy rises above  the required acuracy
required_accuracy = 0.99
loss_function_ids = {0:"Mean Squared Error",1:"Significance Loss",2:"Binery Cross Entropy",3:"Inverted Significance Loss"}

# how to use: set using_full_data to true or false, then pick the loss function id and nubmer of epcohs

def initilize():
    torch.manual_seed(random_seed)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    return network, optimizer



def run_epoch(network,optimizer,loss_function_id,train_loss_list,test_loss_list,fp,tp):


    for batch, (data, target) in enumerate(train_loader):
        if using_full_data == False:
            data,target = subset_data(data,target,background)
        loss = train(network,optimizer,data,target,loss_function_id)
        train_loss_list.append(loss)

    #print(train_loss_list)

    total_number_correct = 0
    total_number = 0
    false_positive_count = 0
    true_positive_count = 0

    for batch, (data, target) in enumerate(test_loader):
        if using_full_data == False:
            data,target = subset_data(data,target,background)
        test_losses,batch_number_correct,batch_true_positive_count,batch_false_positive_count = test(network, data,target,loss_function_id)
        test_loss_list.append(test_losses)
        total_number += target.shape[0]
        total_number_correct += batch_number_correct
        true_positive_count += batch_true_positive_count
        false_positive_count += batch_false_positive_count

    fp.append(false_positive_count)
    tp.append(true_positive_count)

    if using_full_data:
        print("Total number correct :{} ({}%) False positives:{} True Positives{}".format(total_number_correct,100*total_number_correct/mnist_test_size,false_positive_count,true_positive_count))
    else:
        print("Total number correct :{} ({}%) False positives:{} True Positives{}".format(total_number_correct,100*total_number_correct/(mnist_test_size/5),false_positive_count,true_positive_count))
    return network, optimizer, train_loss_list,test_loss_list,total_number_correct,tp,fp

def file_check(path):
    if os.path.exists(path):
        print("A file already exists at {}".format(path))
        overwite = input("Would you like to overwrite it? y/n")
        if overwite == "y":
            return True
        else:
            return False
    else:
        return True


def train_and_test(loss_function_id,experiment_name):



    loss_graph_filepath = "loss_graphs/"  + str(experiment_name) + ".png"
    nn_filepath = "neuralnets/" + str(experiment_name) + ".pth"

    if not file_check(nn_filepath):
        print("quit to protect network file")
        return "quit to protect network file"
    if not file_check(loss_graph_filepath):
        print("quit to protect network file")
        return "quit to protect loss graph"
    if use_auto_stop:
        print("training until {}% acuracy with {}...".format(required_accuracy*100,loss_function_ids[loss_function_id]))
    else:
        print("training for {} epochs with {}...".format(n_epochs,loss_function_ids[loss_function_id]))

    network, optimizer = initilize()
    test_loss_list = []
    train_loss_list = []
    correct = []
    fp = []
    tp = []
    if not use_auto_stop:
        for epoch in range(1, n_epochs + 1):
            print("Epoch number {}".format(epoch))
            network, optimizer, train_loss_list,test_loss_list,correct_per_epoch,tp,fp = run_epoch(network,optimizer,loss_function_id,train_loss_list,test_loss_list,fp,tp)
            correct.append(correct_per_epoch)
    else:
        accuracy = 0
        epoch = 0
        while accuracy < required_accuracy:
            epoch = epoch + 1
            print("Epoch number {}".format(epoch))
            network, optimizer, train_loss_list,test_loss_list,correct_per_epoch,tp,fp = run_epoch(network,optimizer,loss_function_id,train_loss_list,test_loss_list,fp,tp)
            if using_full_data:
                accuracy = correct_per_epoch/mnist_test_size
            else:
                accuracy = correct_per_epoch/(mnist_test_size/5)
            correct.append(correct_per_epoch)



    torch.save(network.state_dict(), nn_filepath)

    font1 = {'size':5}

    plt.subplot(2, 2, 1)
    plt.title("Training", fontdict = font1)

    plt.ylabel("Training Loss (per batch)", fontdict = font1)
    plt.plot(train_loss_list)

    plt.subplot(2, 2, 2)
    plt.title("Testing", fontdict = font1)

    plt.xlabel("Epoch number", fontdict = font1)
    plt.ylabel("Test Loss", fontdict = font1)
    plt.plot(test_loss_list)


    significances = []
    if loss_function_id == 1:
        for i in range(len(test_loss_list)):
            significances.append(math.sqrt(-1 * test_loss_list[i][0]))
    else:
        print("using seperate sig evaluation")
        for i in range(len(tp)):
            significances.append(tp[i]/math.sqrt(tp[i] + fp[i] + 0.00000001))



    plt.subplot(2, 2, 3)
    plt.title("Significance", fontdict = font1)
    plt.xlabel("Epoch number", fontdict = font1)
    plt.ylabel("Testing Significance", fontdict = font1)
    plt.plot(significances)


    plt.subplot(2, 2, 4)
    plt.title("Accuracy", fontdict = font1)
    plt.xlabel("Epoch number", fontdict = font1)
    plt.ylabel("percent accuracy", fontdict = font1)
    for i in range(len(correct)):
        if using_full_data:
            correct[i] = 100*correct[i]/mnist_test_size
        else:
            correct[i] = 100*correct[i]/(mnist_test_size/5)


    plt.plot(correct)


    if use_auto_stop:
        num_epochs  = epoch
    else:
        num_epochs = n_epochs
    if using_full_data:
        title = str(loss_function_ids[loss_function_id]) + " with full dataset for " + str(num_epochs) + " epochs"
    else:
        title = str(loss_function_ids[loss_function_id]) + " with partial dataset for " + str(num_epochs) + " epochs"

    plt.suptitle(str(title),fontdict = font1)
    plt.savefig(loss_graph_filepath)


train_and_test(2,"bce_auto_full") #not a file path, this just the name

