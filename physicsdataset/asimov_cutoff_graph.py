import torch
from physicsdataset.phy_functions import asimov_significance, discrete_asimov_significance
from physicsdataset.phy_net import Net_256_512_512_256, Net_256_512
import matplotlib.pyplot as plt


def compute_plot_asimov_cutoffs(network, systematic_list):
    test_data_path = "../non_normalized_loaded_data/test_data_nb_" + str(12) + "_bs_" + str(4000) + ".pt"
    test_target_path = "../non_normalized_loaded_data/test_target_nb_" + str(12) + "_bs_" + str(4000) + ".pt"

    test_data = torch.load(test_data_path)
    test_target = torch.load(test_target_path)
    cutoff_list = []
    for i in range(41):
        cutoff_list.append(i/40)
    print(cutoff_list)
    print(systematic_list)

    significances_per_systematic = []

    for systematic in systematic_list:
        significances_per_cutoff = []
        for cutoff in cutoff_list:
            #print(cutoff)
            cutoff_significance = 0
            for batch in range(12):
                network.eval()
                data_batch = test_data[batch]
                target_batch = test_target[batch]
                output = network(data_batch)

                significance = discrete_asimov_significance(output, target_batch, cutoff, systematic)
                cutoff_significance += significance
            significances_per_cutoff.append(cutoff_significance)
        significances_per_systematic.append(significances_per_cutoff)
    return significances_per_systematic


def visulize_asimov_cutoffs(network_name, systematic_list):
    network = Net_256_512()
    network.load_state_dict(torch.load("../new_phy_nets/" + network_name +  ".pth"))


    significances_per_systematic = compute_plot_asimov_cutoffs(network, systematic_list)
    print(significances_per_systematic)
    #plt.figure(figsize=(20, 20))

    for systematic_error in range(len(significances_per_systematic)):
        errors_per_cutoff = significances_per_systematic[systematic_error]
        plt.plot(errors_per_cutoff)

    plt.legend(systematic_list, loc ="lower right")
    plt.title("Asimov Estimate of Significance verses cutoff for "+ str(network_name), fontsize = 10)
    plt.savefig("../asimov_evaluation_plots/" + str(network_name)  + ".png")


network_name = "ae_0.001_256_512_0.05"
visulize_asimov_cutoffs(network_name, [0.1,0.3,0.5])
