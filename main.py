import torch
import torch.optim as optim

from net import Net
from test import test
from train import train
from variables import *

torch.backends.cudnn.enabled = False

torch.manual_seed(random_seed)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)




for epoch in range(1, n_epochs + 1):
    print("Epoch number {}".format(epoch))


    train_losses = train(network,optimizer,num_train_batches,train_batch_size,epoch)


    print("TRAINING LOSSES {}".format(train_losses))

    hits,correct, test_losses,num_times_signal_was_missed,num_times_signal_was_contaminated,num_times_signal_appears_in_dataset, false_positive_count,true_positive_count,deviations = test(network,num_test_batches,test_batch_size,-1,True)
    print("TESTING COMPLETE")
    print(deviations)
    print("{} correct, {} loss, {} times signal was missed, {}times signal was contaminated, {} times signal appeared in the data".format(correct,test_losses,num_times_signal_was_missed,num_times_signal_was_contaminated,num_times_signal_appears_in_dataset))

    torch.save(network.state_dict(), '/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/neuralnets/model.pth')




