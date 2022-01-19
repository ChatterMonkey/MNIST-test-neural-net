from physicsdataset.phy_functions import find_loss
from physicsdataset.phy_variables import variables
import torch

def test(network, data, target, weights, loss_function_id,calculating_tp_and_fp = False, cutoff = 0.5):

    network.eval()
    output = network(data)
    loss = find_loss(output,target,weights, loss_function_id)

    num_correct = 0
    guessed_signal = 0
    guessed_bkground = 0

    if calculating_tp_and_fp:
        tp = 0
        fp = 0
    #print(output)
    #print(output[0])
    #print(output[variables.test_batch_size - 1])
    cutoff_tensor = torch.full((variables.test_batch_size, 1), cutoff, device='cuda')
    compare_tensor = torch.gt(output, cutoff_tensor)
    #print(compare_tensor)
    #find true positives
    trup = torch.sum(compare_tensor*target)
    #find false positives
    falp = torch.sum(compare_tensor*(1 - target))
    #find the total number correct
    num_correct_p = torch.sum(torch.eq(compare_tensor,target))

    #print(target)
    #print(1 - target)
    #print(trup)
    #print(falp)
    #print(num_correct_p)




#    for event in range(variables.test_batch_size):
#
#        if output[event][0].item() >= cutoff:
#
#            guessed_signal += 1
#            if target[event][0] == 1:
#                num_correct += 1
#                if calculating_tp_and_fp:
#                    tp += 1
#            else:
#                if calculating_tp_and_fp:
#                    fp += 1
#        else:
#            guessed_bkground += 1
#            if target[event][0] ==0:
#                num_correct += 1
#    print("tp")
#    print(tp)
#    print(fp)
#    print(num_correct)
    if calculating_tp_and_fp:
        #return num_correct,loss, tp,fp
        return num_correct_p, loss, trup, falp
    return num_correct, loss
