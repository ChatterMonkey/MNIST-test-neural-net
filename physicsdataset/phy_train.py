from physicsdataset.phy_functions import find_loss


def train(network, optimizer, data, target, loss_function_id,  s_weight_tracker,b_weight_tracker, weights = None):
    network.train()
    optimizer.zero_grad()
    output = network(data)
    #print("OUTPUT IS {}   {}".format(output,data))
    loss, s_weight_tracker,b_weight_tracker = find_loss(output,target, weights, loss_function_id, s_weight_tracker,b_weight_tracker)
    loss.backward()
    optimizer.step()
    return loss.item(), s_weight_tracker,b_weight_tracker
