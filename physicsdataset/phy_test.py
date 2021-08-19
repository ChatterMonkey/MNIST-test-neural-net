from physicsdataset.phy_functions import find_loss
from physicsdataset.phy_variables import variables


def test(network, data, target, loss_function_id,calculating_tp_and_fp = False, cutoff = 0.5):

    output = network(data)
    loss = find_loss(output,target,loss_function_id)


    num_correct = 0

    if calculating_tp_and_fp:
        tp = 0
        fp = 0

    for event in range(variables.test_batch_size):


        if output[event][0].item() > cutoff:
            if target[event][0]  == 1:
                num_correct += 1
                if calculating_tp_and_fp:
                    tp += 1
            else:
                if calculating_tp_and_fp:
                    fp += 1
        else:
            if target[event][0] ==0:
                num_correct += 1

    if calculating_tp_and_fp:
        return num_correct,loss, tp,fp
    return num_correct, loss
