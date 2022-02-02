from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.phy_net import Net_256_512_512_256, Net_256_512
from physicsdataset.phy_train import train
from physicsdataset.phy_test import test
from physicsdataset.data_manager import add_data, visulize, split_weights_from_target
from physicsdataset.phys_roc_maker import calculate_roc_curve_points
from physicsdataset.phy_variables import variables
from os.path import exists
import os
import json
import torch
import torch.optim as optim
import time
import yappi
torch.manual_seed(1)

device = 'cuda'  # set device as 'cuda' for gpu or 'cpu'
monitor_code = False  # keep track of how long is spent in each function
test_mode = True  # used when running locally
saving_frequency = 20  # how often to save data
loging_frequency = 10  # how often t log epoch number
pvc_path = "data_storage"  # path to persitent volume
loaded_data_path = "weighted_non-normalized_loaded_data"

if monitor_code:
    yappi.set_clock_type('cpu')
    yappi.start()

print("Deploying The Sparkle Squid with {} ...".format(device))
print("")
print("testing mode is on: {}".format(test_mode))


# 250000 total records in the data set
if not test_mode: # load variables from the environment
    loss_function_id = int(os.environ['lossFunctionId'])
    num_epochs = int(os.environ['numEpochs'])
    learning_rate = float(os.environ['learningRate'])
    systematic = float(os.environ['systematicError'])
    num_training_batches = int(os.environ['numTrainingBatches'])
    num_testing_batches = int(os.environ['numTestingBatches'])
    train_batch_size = int(os.environ['trainBatchSize'])
    test_batch_size = int(os.environ['testBatchSize'])
    test_nickname = os.environ['testNote']
else:  # set variables manually for testing
    loss_function_id = 3 # (("mean squared error","mse"),("significance loss","sl"),("binery cross entropy","bce"),("asimov estimate","ae"),("inverted significance loss","isl"))
    num_epochs = 1000
    learning_rate = 0.001
    systematic = 0.1
    num_training_batches = 50
    num_testing_batches = 12
    train_batch_size = 4000
    test_batch_size = 4000
    use_weights = False
    test_nickname = "test"

variables.set_params(device, train_batch_size, test_batch_size, num_training_batches, num_testing_batches, loss_function_id,learning_rate, num_epochs, systematic)
print("systematic is {}".format(variables.systematic))


# create test name
if use_weights:
    test_name = variables.loss_function_tuple[loss_function_id][1] + "_" + str(learning_rate) + "_" + "weights_" + test_nickname
else:
    test_name = variables.loss_function_tuple[loss_function_id][1] + "_" + str(
        learning_rate) + "_" + "noweights_" + test_nickname
print("test name: {}".format(test_name))

# how many local minimums are allowed in the loss before training is stopped
minimums_allowed = 5
minimums_remaining = minimums_allowed


network_path = pvc_path + "/" + test_name + ".pth"
best_network_path = pvc_path + "/" + test_name + "_best" + ".pth"
plot_path = pvc_path + "/" + test_name + ".png"
data_path = pvc_path + "/" + test_name + "_data.json"

network = Net_256_512()
network = network.to(device)

print("there exists a previous network to start from: {}".format(exists(network_path)))  # check if a network already exists to start from
if exists(network_path):
    print("Initializing with older network")
    network.load_state_dict(torch.load(network_path))
optimizer = optim.Adam(network.parameters(), learning_rate)


training_data_path = loaded_data_path + "/train_data_nb_" + str(num_training_batches) + "_bs_" + str(
    variables.train_batch_size) + ".pt"
training_target_path = loaded_data_path +  "/train_target_nb_" + str(num_training_batches) + "_bs_" + str(
    variables.train_batch_size) + ".pt"
test_data_path = loaded_data_path + "/test_data_nb_" + str(num_testing_batches) + "_bs_" + str(
    variables.test_batch_size) + ".pt"
test_target_path = loaded_data_path + "/test_target_nb_" + str(num_testing_batches) + "_bs_" + str(
    variables.test_batch_size) + ".pt"

if exists(training_data_path) and exists(training_target_path):
    print("Using preloaded training data")
    train_data = torch.load(training_data_path)
    train_target_and_weights = torch.load(training_target_path)

else:
    train_data, train_target_and_weights = open_training_data("non_normalized_loaded_data", num_training_batches)


if exists(test_data_path) and exists(test_target_path):
    print("Using preloaded testing data")
    test_data = torch.load(test_data_path)
    test_target_and_weights = torch.load(test_target_path)
else:
    test_data, test_target_and_weights = open_test_data("non_normalized_loaded_data", num_testing_batches)
train_data = train_data.to(device)
train_target_and_weights = train_target_and_weights.to(device)
test_data = test_data.to(device)
test_target_and_weights = test_target_and_weights.to(device)
print(train_target_and_weights.device)
train_target, train_weights = split_weights_from_target(train_target_and_weights)
test_target, test_weights = split_weights_from_target(test_target_and_weights)
print("number of signal events")
print(torch.sum(train_target))
print(train_target.shape)


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

previous_epochs_completed = len(accuracy_each_epoch)
num_epochs = num_epochs - previous_epochs_completed # determine how many more epochs must be run
variables.restore_epochs(num_epochs)
print("{} more epochs will be run".format(variables.num_epochs))


running_count_of_epochs_needed_to_train = previous_epochs_completed


loss_moving_average = 0
all_time_low_loss = None
loss_is_minimized = False
loss_reached_local_minimum = True

loss_is_leq = True


def save_data(data_path, accuracy_each_epoch, training_loss_each_epoch, testing_loss_each_epoch, tp_overtime,
              fp_overtime):
    with open(data_path, 'w') as file:
        data = {
            "accuracy_each_epoch": accuracy_each_epoch,
            "training_loss_each_epoch": training_loss_each_epoch,
            "testing_loss_each_epoch": testing_loss_each_epoch,
            "tp_overtime": tp_overtime,
            "fp_overtime": fp_overtime
        }
        data_dumps = json.dumps(data)
        file.write(data_dumps)
    with open(data_path, 'r') as file:
        print(file.read())
        return


def save_network(network, network_path):
    torch.save(network.state_dict(), network_path)
    return


for epoch in range(variables.num_epochs):
    if epoch % 10 == 0:
        print("epoch {} of {} this restart, so far has taken {} epochs".format(epoch, variables.num_epochs, running_count_of_epochs_needed_to_train))
    if not loss_is_minimized:
        running_count_of_epochs_needed_to_train += 1

        num_correct_this_epoch = 0  # sum across all of the batches in an epoch
        training_loss_this_epoch = 0
        testing_loss_this_epoch = 0

        for batch in range(num_training_batches):
            batch_train_data = train_data[batch]
            batch_train_target = train_target[batch]

            if use_weights:
                batch_train_weights = train_weights[batch]
                loss = train(network, optimizer, batch_train_data, batch_train_target, loss_function_id, weights = batch_train_weights )
            else:
                loss = train(network, optimizer, batch_train_data, batch_train_target, loss_function_id)

            training_loss_this_epoch += loss
        training_loss_each_epoch.append(training_loss_this_epoch)
        epoch_tp = 0
        epoch_fp = 0
        for batch in range(num_testing_batches):

            batch_test_data = test_data[batch]
            batch_test_target = test_target[batch]

            if use_weights:
                batch_test_weights = test_weights[batch]

                num_correct, loss, tp, fp = test(network, batch_test_data, batch_test_target ,loss_function_id,
                                             calculating_tp_and_fp=True, weights = batch_test_weights)
            else:
                num_correct, loss, tp, fp = test(network, batch_test_data, batch_test_target, loss_function_id,
                                                 calculating_tp_and_fp=True)

            epoch_tp += tp

            testing_loss_this_epoch += loss.item()
            num_correct_this_epoch += num_correct

        if epoch == 0:
            all_time_low_loss = testing_loss_this_epoch
            print("setting up all time low = {}".format(testing_loss_this_epoch))
        else:
            if epoch % saving_frequency != 0:
                # print("intermediate epoch, adding {}".format(testing_loss_this_epoch/19))
                loss_moving_average += testing_loss_this_epoch / (saving_frequency - 1)
            else:
                print("multiple of {}!, all time low loss is {} and average is {}, saving".format(saving_frequency, all_time_low_loss,
                                                                                                  loss_moving_average))
                save_data(data_path, accuracy_each_epoch, training_loss_each_epoch, testing_loss_each_epoch,
                          true_positives_over_time, false_positives_over_time)
                save_network(network, network_path)
                if (loss_moving_average < all_time_low_loss) or (loss_moving_average == all_time_low_loss):
                    loss_is_leq = True
                else:
                    loss_is_leq = False
                if loss_is_leq:
                    all_time_low_loss = loss_moving_average
                    loss_moving_average = 0
                    minimums_remaining = minimums_allowed
                if not loss_is_leq:
                    if minimums_remaining == minimums_allowed:
                        print("SAVED NET!!!!!!")
                        torch.save(network.state_dict(), best_network_path)
                    print("loss moving average {} all time low {}".format(loss_moving_average, all_time_low_loss))

                    if minimums_remaining > 0:
                        minimums_remaining = minimums_remaining - 1
                        print("loss is locally minimized, minimum {}".format(minimums_remaining))
                    else:
                        loss_is_minimized = True
                        print("loss is globally minimized")
                    loss_moving_average = 0

        true_positives_over_time.append(epoch_tp)
        false_positives_over_time.append(epoch_fp)

        testing_loss_each_epoch.append(testing_loss_this_epoch)
        accuracy_each_epoch.append(num_correct_this_epoch / (num_testing_batches * test_batch_size) * 100)
    if epoch == variables.num_epochs - 1:
        print("RAN OUT OF EPOCHS")
        save_data(data_path, accuracy_each_epoch, training_loss_each_epoch, testing_loss_each_epoch,
                  true_positives_over_time, false_positives_over_time)
        save_network(network, network_path)

print("{} epochs needed".format(running_count_of_epochs_needed_to_train))

cutoffs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]
if use_weights:
    tp, fp = calculate_roc_curve_points(cutoffs, network, loss_function_id, test_data, test_target, test_weights)
else:
    tp, fp = calculate_roc_curve_points(cutoffs, network, loss_function_id, test_data, test_target, test_weights)

add_data(network_path, training_loss_each_epoch, testing_loss_each_epoch, accuracy_each_epoch, true_positives_over_time,
         false_positives_over_time, running_count_of_epochs_needed_to_train)
print(tp)
visulize(plot_path, plot_last=True, test_data=test_data, test_target = test_target, weights=test_weights )
print("{} correct, {}% accuracy".format(accuracy_each_epoch[-1], accuracy_each_epoch[-1]))

if monitor_code:
    yappi.get_func_stats().print_all()