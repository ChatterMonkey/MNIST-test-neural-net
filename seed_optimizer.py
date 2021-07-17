import torch
from net import Net

ave_deviations = []
i = 7
deviations = []
torch.manual_seed(i)
net = Net()
test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output = test(net)
for j in range(len(sample_output)):
    average = sum(sample_output[j])/10
    for k in range(10):
        deviation = abs(sample_output[j][k] - average)
        deviations.append(deviation)
print("DEVIATIONS = {}".format(deviations))
print("average deviations = {}".format(sum(deviations)/len(deviations)))
ave_deviations.append(sum(deviations)/len(deviations))
print(ave_deviations)
