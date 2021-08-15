import torch
from physicsdataset.phy_net import Net
from tqdm import tqdm
from physicsdataset.phy_loaders import open_test_data
from physicsdataset.phy_variables import variables
from physicsdataset.phy_test import test

def load_network(path):
    network_state = torch.load(path)
    network = Net()
    network.load_state_dict(network_state)
    return network


def calculate_roc_curve_points(cutoffs, network,loss_function_id,num_testing_batches):

    false_positives = []
    true_positives = []

    test_data_lists,test_target_lists = open_test_data(num_testing_batches)
    for cutoff in enumerate(cutoffs):
        for batch in tqdm(range(num_testing_batches), colour ="blue", desc="Generating ROC curve"):

            test_data_batch = test_data_lists[batch]
            test_target_batch = test_target_lists[batch]
            target_size = len(test_target_batch[0])
            data_size = len(test_data_batch[0])


            target_t = torch.zeros([target_size, 1])
            data_t = torch.zeros([data_size,num_testing_batches])

            for event in range(num_testing_batches):
                target_t[event][0] = test_target_batch[event]

            for event in range(num_testing_batches):
                for variable in range(variables.num_variables):
                    data_t[event][variable] = abs(float(test_data_batch[event][variable])/variables.normalization_constant)
            num_correct,loss, tp,fp = test(network,data_t,target_t,loss_function_id,True,cutoff)

            true_positives.append(tp/data_size)
            false_positives.append(fp/data_size)
    return true_positives,false_positives




