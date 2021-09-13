from physicsdataset.phy_functions import find_loss
from physicsdataset.phy_variables import variables


def test(network, data, target, loss_function_id,calculating_tp_and_fp = False, cutoff = 0.5):
    #print("testing cutoff is {}".format(cutoff))
    #print(variables.test_batch_size)
    #print(target.size())



    output = network(data)

    #print(output)
    #print(target)
    loss = find_loss(output,target,loss_function_id)



    num_correct = 0
    guessed_signal = 0
    guessed_bkground = 0


    if calculating_tp_and_fp:
        tp = 0
        fp = 0

    for event in range(variables.test_batch_size):
        #print(output[event][0].item())
        #print(target[event][0])


        if output[event][0].item() >= cutoff:

            guessed_signal += 1
            if target[event][0] == 1:
                #print("CORRECT")
                num_correct += 1
                if calculating_tp_and_fp:
                    tp += 1
            else:
                if calculating_tp_and_fp:
                    fp += 1
        else:
            guessed_bkground += 1
            if target[event][0] ==0:
                num_correct += 1
    #print("TP AND FP")
    #print(tp)
    #print(fp)
    #print("correct: {}".format(num_correct))
    #print("bk: {}".format(guessed_bkground))
    #print("sig: {}".format(guessed_signal))
    #print("tp: {}".format(tp))
    #print("fp: {}".format(fp))

    if calculating_tp_and_fp:
        return num_correct,loss, tp,fp
    return num_correct, loss
