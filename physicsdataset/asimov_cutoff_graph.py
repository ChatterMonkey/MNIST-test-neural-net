import torch
from physicsdataset.phy_functions import asimov_significance, discrete_asimov_significance


def plot_asimov_cutoffs(network, systematic_list):
    test_data_path = "../non_normalized_loaded_data/test_data_nb_" + str(12) + "_bs_" + str(4000) + ".pt"
    test_target_path = "../non_normalized_loaded_data/test_target_nb_" + str(12) + "_bs_" + str(4000) + ".pt"

    test_data = torch.load(test_data_path)
    test_target = torch.load(test_target_path)
    cutoff_list = []
    for i in range(11):
        cutoff_list.append(i/10)

    significances_per_systematic = []

    for systematic in systematic_list:
        significances_per_cutoff = []
        for cutoff in cutoff_list:
            cutoff_significance = 0
            for batch in range(12):
                network.eval()
                data_batch = test_data[batch]
                target_batch = test_target[batch]
                output = network(data_batch)
                significance = discrete_asimov_significance(output,target_batch,cutoff, systematic)
                cutoff_significance += significance
            significances_per_cutoff.append(cutoff_significance)
        significances_per_systematic.append(significances_per_cutoff)
    print(significances_per_systematic)



