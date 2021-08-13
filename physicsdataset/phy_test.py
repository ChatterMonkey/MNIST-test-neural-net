from physicsdataset.phy_functions import find_loss
from physicsdataset.phy_variables import variables


def test(network, data, target, loss_function_id):

    output = network(data)
    loss = find_loss(output,target,loss_function_id)

    num_correct = 0
    for guess in range(variables.test_batch_size):
        if output[guess][0] > 0.5:
            if target[guess][0]  == 1:
                num_correct += 1
        else:
            if target[guess][0] ==0:
                num_correct += 1
    return num_correct, loss
