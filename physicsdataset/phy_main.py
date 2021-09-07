from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.phy_net import Net
from physicsdataset.phy_train import train
from physicsdataset.phy_test import test
from physicsdataset.data_manager import add_data,visulize
from physicsdataset.phys_roc_maker import calculate_roc_curve_points
from os.path import exists
import torch
import torch.optim as optm
from tqdm import tqdm
from physicsdataset.phy_variables import variables

#sk learn roc_curve y_true, y_score
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# 250000

loss_function_id = 3
num_epochs =800
learning_rate = 0.001

num_training_batches = 50
num_testing_batches = 12

train_batch_size = 4000
test_batch_size = 4000

patience = 20
test_name = input("Test Name:")
minimums_allowed= 4
minimums_occured = minimums_allowed

variables.set_params(train_batch_size,test_batch_size,num_training_batches,num_testing_batches,loss_function_id,learning_rate,num_epochs)

network_path= "../phy_nets/" + test_name + ".pth"
plot_path = '../physics_graphs/'+ test_name + ".png"
torch.manual_seed(1)
network = Net()
network.load_state_dict(torch.load("../phy_nets/preisaac.pth"))

optimizer = optm.Adam(network.parameters(),learning_rate)

#check if data exists:

training_data_path = "../loaded_data/train_data_nb_" + str(num_training_batches) + "_bs_" + str(variables.train_batch_size) + ".pt"
training_target_path = "../loaded_data/train_target_nb_" + str(num_training_batches) + "_bs_" + str(variables.train_batch_size) + ".pt"

test_data_path = "../loaded_data/test_data_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"
test_target_path = "../loaded_data/test_target_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"



if exists(training_data_path) and exists(training_target_path):
    print("Using preloaded training data")
    train_data = torch.load(training_data_path)
    train_target = torch.load(training_target_path)
else:
    train_data, train_target = open_training_data(num_training_batches)

if exists(test_data_path) and exists(test_target_path):
    print("Using preloaded testing data")
    test_data = torch.load(test_data_path)
    test_target = torch.load(test_target_path)
else:
    test_data, test_target = open_test_data(num_testing_batches)




accuracy_each_epoch = []
training_loss_each_epoch = []
testing_loss_each_epoch = []
true_positives_over_time = []
false_positives_over_time = []

loss_moving_average = 0
all_time_low_loss = None
loss_is_minimized = False
loss_reached_local_minimum = True
needed_epochs = 0
loss_is_leq = True


for epoch in range(variables.num_epochs):


    if not loss_is_minimized:
        needed_epochs += 1

        num_correct_this_epoch = 0
        training_loss_this_epoch = 0
        testing_loss_this_epoch = 0

        for batch in range(num_training_batches):

            batch_train_data = train_data[batch]
            batch_train_target = train_target[batch]
            loss = train(network,optimizer,batch_train_data,batch_train_target,loss_function_id)
            training_loss_this_epoch += loss

        training_loss_each_epoch.append(training_loss_this_epoch)
        epoch_tp = 0
        epoch_fp = 0
        for batch in range(num_testing_batches):
            batch_test_data = test_data[batch]
            batch_test_target= test_target[batch]

            num_correct, loss,tp,fp = test(network,batch_test_data,batch_test_target,loss_function_id,calculating_tp_and_fp=True)
            epoch_tp += tp
            epoch_fp += fp


            testing_loss_this_epoch += loss.item()
            num_correct_this_epoch += num_correct
        #if patience > 0 or patience == 0:
            #if loss_reached_local_minimum:
             #   patience = patience -1
        if epoch == 0:
            all_time_low_loss = testing_loss_this_epoch
            print("setting up all time low = {}".format(testing_loss_this_epoch))
        else:
            if epoch%20 != 0:
                #print("intermediate epoch, adding {}".format(testing_loss_this_epoch/19))
                loss_moving_average += testing_loss_this_epoch/19
            else:
                print("multiple of 20!, all time low loss is {} and average is {}".format(all_time_low_loss,loss_moving_average))
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
                        torch.save(network.state_dict(),network_path)
                    print("loss moving average {} all time low {}".format(loss_moving_average,all_time_low_loss))

                    if minimums_occured > 0:
                        minimums_occured = minimums_occured - 1
                        print("loss is locally minimized, minimum {}".format(minimums_occured))
                    else:
                        loss_is_minimized = True
                        print("loss is globally minimized")



        true_positives_over_time.append(epoch_tp)
        false_positives_over_time.append(epoch_fp)

        testing_loss_each_epoch.append(testing_loss_this_epoch)
        accuracy_each_epoch.append(num_correct_this_epoch/(num_testing_batches*test_batch_size)*100)
    if epoch == variables.num_epochs - 1:
        print("RAN OUT OF EPOCHS")
        torch.save(network.state_dict(),network_path)



print("{} epochs needed".format(needed_epochs))


cutoffs= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1]

tp,fp = calculate_roc_curve_points(cutoffs,network,loss_function_id,test_data,test_target)


add_data(network_path,training_loss_each_epoch,testing_loss_each_epoch, accuracy_each_epoch,true_positives_over_time,false_positives_over_time,needed_epochs)
print(tp)

visulize(plot_path,plot_last=True,test_data = test_data,test_target=test_target)

print("{} correct, {}% accuracy".format(accuracy_each_epoch[-1], accuracy_each_epoch[-1] ))









