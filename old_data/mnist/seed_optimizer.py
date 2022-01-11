import torch
import numpy
from old_data.mnist.net import Net
from old_data.mnist.test import test

def optimize_seed(numberofseeds):
    ave_deviations_per_seed = []
    for i in range(numberofseeds):
        ave_deviations_per_query = []
        torch.manual_seed(i)
        net = Net()
        test_losses, total_number_correct, true_positive_count, false_positive_count, sample_output = test(net)
        absolute_deviations = []
        for j in range(len(sample_output)):
            query_average = sum(sample_output[j]) / 10
            for k in range(10):
                absolute_deviation = abs(sample_output[j][k] - query_average)
                absolute_deviations.append(absolute_deviation)
        print("DEVIATIONS = {}".format(absolute_deviation))
        print("average deviations for each query  = {}".format(sum(absolute_deviations) / len(absolute_deviations)))
        ave_deviations_per_seed.append(sum(absolute_deviations) / len(absolute_deviations))
    print(ave_deviations_per_seed)
    print("the best seed is {}".format(numpy.argmin(ave_deviations_per_seed)))
