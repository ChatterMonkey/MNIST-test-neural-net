from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.phy_net import Net_256_512_512_256, Net_256_512
from physicsdataset.phy_train import train
from physicsdataset.phy_test import test
from physicsdataset.data_manager import add_data,visulize,split_weights_from_target
from physicsdataset.phys_roc_maker import calculate_roc_curve_points
from os.path import exists
import os
import json
import torch
import torch.optim as optm
import numpy as np
from tqdm import tqdm
from physicsdataset.phy_variables import variables

pvc_path = "mock_task-pv-claim"
loaded_data_path = "/weighted_non-normalized_loaded_data"

# 250000 total records
#loss_function_id = int(os.environ['lossFunctionId'])
#num_epochs = int(os.environ['numEpochs'])
#learning_rate = float(os.environ['learningRate'])
#systematic = float(os.environ['systematicError'])

#num_training_batches = int(os.environ['numTrainingBatches'])
#num_testing_batches = int(os.environ['numTestingBatches'])

#train_batch_size = int(os.environ['trainBatchSize'])
#test_batch_size = int(os.environ['testBatchSize'])

#test_nickname = os.environ['testNote']
#mse = 0,sl = 1,bce = 2,ae = 3,isl = 4

loss_function_id = 3
num_epochs = 500
learning_rate = 0.001
systematic = 0

num_training_batches = 50
num_testing_batches = 12

train_batch_size = 4000
test_batch_size = 4000

test_note = "**"
print(test_note)

test_name = variables.loss_function_tuple[loss_function_id][1] + "_" + str(learning_rate) + "_" + test_note
print(test_name)

minimums_allowed= 5
minimums_occured = minimums_allowed

variables.set_params(train_batch_size,test_batch_size,num_training_batches,num_testing_batches,loss_function_id,learning_rate,num_epochs,systematic)

#network_path= "new_phy_nets/" + test_name + ".pth"
#plot_path = 'new_phy_graphs/'+ test_name + ".png"

network_path = "../" + pvc_path + "/" + test_name + ".pth"
network_path2 = "../" + pvc_path + "/" + test_name + "_best" + ".pth"
plot_path = "../" + pvc_path + "/" + test_name + ".png"
data_path = "../" + pvc_path + "/" + test_name  + "_data.json"
#
#network_path = pvc_path + "/" + test_name + ".pth"
#best_network_path = pvc_path + "/" + test_name + "_best" + ".pth"
#plot_path = pvc_path + "/" + test_name + ".png"
#data_path =  "/app/" + pvc_path + "/" + test_name + "_data.json"
#data_path2 =   pvc_path + "/" + test_name + "_data.json"


torch.manual_seed(1)
network = Net_256_512()
print(exists(network_path))
if exists(network_path):
    print("Initilizing with older network")
    network.load_state_dict(torch.load(network_path))
optimizer = optm.Adam(network.parameters(),learning_rate)

#training_data_path = "../non_normalized_loaded_data/train_data_nb_" + str(num_training_batches) + "_bs_" + str(variables.train_batch_size) + ".pt"
#training_target_path = "../non_normalized_loaded_data/train_target_nb_" + str(num_training_batches) + "_bs_" + str(variables.train_batch_size) + ".pt"
training_data_path = loaded_data_path + "/train_data_nb_" + str(num_training_batches) + "_bs_" + str(variables.train_batch_size) + ".pt"
training_target_path = loaded_data_path + "/train_target_nb_" + str(num_training_batches) + "_bs_" + str(variables.train_batch_size) + ".pt"

#test_data_path = "../non_normalized_loaded_data/test_data_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"
#test_target_path = "../non_normalized_loaded_data/test_target_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"
test_data_path = loaded_data_path + "/test_data_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"
test_target_path = loaded_data_path + "/test_target_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"

print(exists(test_data_path))
if exists(training_data_path) and exists(training_target_path):
    print("Using preloaded training data")
    train_data = torch.load(training_data_path)
    train_target_and_weights = torch.load(training_target_path)
else:
    train_data, train_target_and_weights = open_training_data(loaded_data_path, num_training_batches)

if exists(test_data_path) and exists(test_target_path):
    print("Using preloaded testing data")
    test_data = torch.load(test_data_path)
    test_target_and_weights = torch.load(test_target_path)
else:
    test_data, test_target_and_weights = open_test_data(loaded_data_path, num_testing_batches)

#split target from weights

train_target, train_weights = split_weights_from_target(train_target_and_weights)
test_target, test_weights = split_weights_from_target(test_target_and_weights)


if exists(data_path):
    print("Backing up from old data...")
    with open(data_path, "r") as file:
        data_text = file.read()
        data = json.loads(data_text)
        accuracy_each_epoch = data['accuracy_each_epoch']
        training_loss_each_epoch = data['training_loss_each_epoch']
        testing_loss_each_epoch = data['testing_loss_each_epoch']
        true_positives_over_time = data['tp_overtime']
        false_positives_over_time = data['fp_overtime']
else:
    accuracy_each_epoch = []
    training_loss_each_epoch = []
    testing_loss_each_epoch = []
    true_positives_over_time = []
    false_positives_over_time = []

num_epochs = num_epochs - len(accuracy_each_epoch)
print(num_epochs)
variables.restore_epochs(num_epochs)
print(variables.num_epochs)

loss_moving_average = 0
all_time_low_loss = None
loss_is_minimized = False
loss_reached_local_minimum = True
needed_epochs = 0
loss_is_leq = True


def save_data(data_path, accuracy_each_epoch, training_loss_each_epoch, testing_loss_each_epoch, tp_overtime, fp_overtime):
    #if not os.path.exists(data_path):
    with open(data_path,'w') as file:

        data = {
            "accuracy_each_epoch": accuracy_each_epoch,
            "training_loss_each_epoch": training_loss_each_epoch,
            "testing_loss_each_epoch": testing_loss_each_epoch,
            "tp_overtime": tp_overtime,
            "fp_overtime": fp_overtime
        }
        data_dumps = json.dumps(data)
        file.write(data_dumps)
    with open(data_path,'r') as file:
        print(file.read())
        return


def save_network(network, network_path):
    torch.save(network.state_dict(),network_path)
    return



for epoch in range(variables.num_epochs):
    print("epoch {} of {}".format(epoch,variables.num_epochs))

    if not loss_is_minimized:
        needed_epochs += 1

        num_correct_this_epoch = 0
        training_loss_this_epoch = 0
        testing_loss_this_epoch = 0

        for batch in range(num_training_batches):

            batch_train_data = train_data[batch]
            batch_train_target = train_target[batch]
            batch_train_weights = train_weights[batch]



            loss = train(network,optimizer,batch_train_data,batch_train_target,batch_train_weights,loss_function_id)
            training_loss_this_epoch += loss
        training_loss_each_epoch.append(training_loss_this_epoch)
        epoch_tp = 0
        epoch_fp = 0
        for batch in range(num_testing_batches):
            batch_test_data = test_data[batch]
            batch_test_weights = test_weights[batch]
            batch_test_target = test_target[batch]

            num_correct, loss,tp,fp = test(network,batch_test_data,batch_test_target,batch_test_weights, loss_function_id,calculating_tp_and_fp=True)
            epoch_tp += tp
            epoch_fp += fp

            testing_loss_this_epoch += loss.item()
            num_correct_this_epoch += num_correct

        if epoch == 0:
            all_time_low_loss = testing_loss_this_epoch
            print("setting up all time low = {}".format(testing_loss_this_epoch))
        else:
            if epoch%20 != 0:
                #print("intermediate epoch, adding {}".format(testing_loss_this_epoch/19))
                loss_moving_average += testing_loss_this_epoch/19
            else:
                print("multiple of 20!, all time low loss is {} and average is {}, saving".format(all_time_low_loss,loss_moving_average))
                save_data(data_path,accuracy_each_epoch,training_loss_each_epoch,testing_loss_each_epoch,true_positives_over_time,false_positives_over_time)
                save_network(network,network_path)
                if (loss_moving_average < all_time_low_loss) or (loss_moving_average == all_time_low_loss):
                    loss_is_leq = True
                else:
                    loss_is_leq = False
                if loss_is_leq:
                    all_time_low_loss = loss_moving_average
                    loss_moving_average = 0
                    minimums_occured = minimums_allowed
                if not loss_is_leq:
                    if minimums_occured == minimums_allowed:
                        print("SAVED NET!!!!!!")
                        torch.save(network.state_dict(),network_path2)
                    print("loss moving average {} all time low {}".format(loss_moving_average,all_time_low_loss))

                    if minimums_occured > 0:
                        minimums_occured = minimums_occured - 1
                        print("loss is locally minimized, minimum {}".format(minimums_occured))
                    else:
                        loss_is_minimized = True
                        print("loss is globally minimized")
                    loss_moving_average = 0

        true_positives_over_time.append(epoch_tp)
        false_positives_over_time.append(epoch_fp)

        testing_loss_each_epoch.append(testing_loss_this_epoch)
        accuracy_each_epoch.append(num_correct_this_epoch/(num_testing_batches*test_batch_size)*100)
    if epoch == variables.num_epochs - 1:
        print("RAN OUT OF EPOCHS")
        save_data(data_path,accuracy_each_epoch,training_loss_each_epoch,testing_loss_each_epoch,true_positives_over_time,false_positives_over_time)
        save_network(network,network_path)

print("{} epochs needed".format(needed_epochs))

cutoffs= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1]
tp,fp = calculate_roc_curve_points(cutoffs, network, loss_function_id, test_data, test_target, test_weights)
add_data(network_path,training_loss_each_epoch,testing_loss_each_epoch, accuracy_each_epoch,true_positives_over_time,false_positives_over_time,needed_epochs)
print(tp)
visulize(plot_path, plot_last=True, test_data = test_data, test_target_and_weights=test_target_and_weights)
print("{} correct, {}% accuracy".format(accuracy_each_epoch[-1], accuracy_each_epoch[-1] ))









