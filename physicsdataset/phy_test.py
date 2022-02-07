from physicsdataset.phy_functions import find_loss
from physicsdataset.phy_variables import variables
import torch


def test(network, data, target, loss_function_id,signal_batch_weight_tracker,background_batch_weight_tracker, calculating_tp_and_fp = False, cutoff = 0.5, weights = None):

    network.eval()
    output = network(data)
    loss, signal_batch_weight_tracker,background_batch_weight_tracker = find_loss(output,target,weights, loss_function_id, signal_batch_weight_tracker,background_batch_weight_tracker)
    cutoff_tensor = torch.full((variables.test_batch_size, 1), cutoff, device=variables.device)
    compare_tensor = torch.gt(output, cutoff_tensor)
    true_positive_count = torch.sum(compare_tensor*target)
    false_positive_count = torch.sum(compare_tensor*(1 - target))
    total_number_correct = torch.sum(torch.eq(compare_tensor,target))
    if calculating_tp_and_fp:
        return total_number_correct.item(), loss, true_positive_count.item(), false_positive_count.item(), signal_batch_weight_tracker,background_batch_weight_tracker
    return total_number_correct.item(), loss, signal_batch_weight_tracker,background_batch_weight_tracker
