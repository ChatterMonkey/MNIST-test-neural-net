from physicsdataset.phy_functions import find_loss


def train(network, optimizer, data, target, loss_function_id):
    network.train()
    optimizer.zero_grad()


    output = network(data)
    loss = find_loss(output,target,loss_function_id)
    loss.backward()
    optimizer.step()
    return loss.item()
