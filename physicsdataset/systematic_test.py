from physicsdataset.phy_functions import discrete_asimov_significance
from physicsdataset.phy_net import Net_256_512,Net_256_512_512_256
import torch
import matplotlib.pyplot as plt


test_data_path = "../non_normalized_loaded_data/test_data_nb_" + str(12) + "_bs_" + str(4000) + ".pt"
test_target_path = "../non_normalized_loaded_data/test_target_nb_" + str(12) + "_bs_" + str(4000) + ".pt"

test_data = torch.load(test_data_path)
test_target = torch.load(test_target_path)

def find_best_asimov_significance(net_name,test_data,test_target):
    network = Net_256_512()
    network.load_state_dict(torch.load("../new_phy_nets/"+str(net_name)+".pth"))
    cutoffs = []
    for i in range(11):
        cutoffs.append(i/10)


    significance_per_cutoff = []

    for cutoff in cutoffs:
        cutoff_significance = 0
        for batch in range(12):
            network.eval()
            data_batch = test_data[batch]
            target_batch = test_target[batch]
            output = network(data_batch)

            significance = discrete_asimov_significance(output, target_batch, cutoff, 0.1)
            cutoff_significance += significance
        significance_per_cutoff.append(cutoff_significance)
    max_significance = 0
    cutoff = 0
    for position, significance_i in enumerate(significance_per_cutoff):
        if significance_i > max_significance:
            max_significance = significance_i
            cutoff = cutoffs[position]
    print("Maximum significance obtained for {} is {} at {}".format(net_name,max_significance,cutoff))

find_best_asimov_significance("ae_0.001_256_512_1.0",test_data,test_target)


def plot_train_significances(test_significance_list,systematic_list):
    plt.plot(systematic_list, test_significance_list)
    plt.xlabel("Systematic Error used during Training")
    plt.ylabel("Asimov estimate of significance for 0.5 sys error")
    plt.title("Maximum Testing significance achieved verses training significance")
    for x,y in zip(systematic_list, test_significance_list):
        label = "{:.3f}".format(x)

        plt.annotate(label, (x,y),fontsize="x-small")
    plt.savefig("../significance_tests/significance_test.png")



#systematic_list = [0,0.05,0.075,0.1,0.15,0.175,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#significance_list = [32.12102513,88.94427347,100.4022109, 103.2337381,102.52489,92.62378958, 90.4569545,78.03975411,81.39058677,67.98789907,74.18249938,77.94990836,77.94990836,78.07055523,78.07055523]
#plot_train_significances(significance_list,systematic_list)
