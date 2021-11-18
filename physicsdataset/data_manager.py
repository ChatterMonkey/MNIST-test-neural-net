import sqlite3 as sq
import json
from physicsdataset.phy_variables import variables as v
from physicsdataset.phys_roc_maker import calculate_roc_curve_points
from physicsdataset.phy_net import Net_256_512_512_256, Net_256_512
from physicsdataset.phy_functions import asimov_from_tp_fp
import matplotlib.pyplot as plt
from physicsdataset.phy_loaders import open_test_data
import math
import torch
import numpy as np


connection = sq.connect("data.sql")
cur = connection.cursor()
e = cur.execute

#e('DROP TABLE data')

def split_weights_from_target(target_and_weights):
    print("Splitting weights and target")
    print(target_and_weights.size())
    target, weights = np.split(target_and_weights, 2,axis = 2)
    return target, weights

def add_data(network_path, training_loss,testing_loss,accuracy,tp_list,fp_list,num_epochs):
    network_string = json.dumps(network_path)

    trainlj = json.dumps(training_loss)

    testlj = json.dumps(testing_loss)
    accj = json.dumps(accuracy)
    tp = json.dumps(tp_list)
    fp = json.dumps(fp_list)

    e('CREATE TABLE IF NOT EXISTS data(ID INTEGER PRIMARY KEY AUTOINCREMENT, network, train_batch_size, test_batch_size, num_training_batches, num_testing_batches, loss_function_id, learning_rate, num_epochs,training_loss,testing_loss, accuracy,tp,fp)')
    e('INSERT INTO data (network, train_batch_size, test_batch_size, num_training_batches, num_testing_batches, loss_function_id, learning_rate, num_epochs,training_loss,testing_loss, accuracy,tp,fp) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(network_string,v.train_batch_size,v.test_batch_size,v.num_training_batches,v.num_testing_batches,v.loss_function_id,v.learning_rate,num_epochs, trainlj,testlj,accj,tp,fp))
    #print(e('select * from data').fetchall())

def visulize(plot_path,experiment_id = 1, plot_last = False,test_data = None,test_target_and_weights = None):
    #print("visulizing")

    cur = connection.cursor()
    if plot_last:
        data = cur.execute('SELECT * FROM data ORDER BY id DESC LIMIT 1').fetchall()

    else:
        data = cur.execute('SELECT * FROM data WHERE id = ?',(experiment_id,)).fetchall()

    data = data[0]

    #print("id , train_batch_size, test_batch_size, num_training_batches, num_testing_batches, loss_function_id, learning_rate, num_epochs,training_loss,testing_loss, accuracy,tp,fp")

    network = Net_256_512()
    network_state_dict = torch.load(json.loads(data[1]))
    network.load_state_dict(network_state_dict)
    loss_function_id = data[6]



    loss_function_str = v.loss_function_tuple[data[6]][0]
    learning_rate = data[7]
    num_epochs = data[8]
    num_training_batches = data[4]
    train_batch_size = data[2]
    num_test_batches = data[5]
    test_batch_size = data[3]

    accuracy = json.loads(data[11])

    tp = json.loads(data[12])
    fp = json.loads(data[13])



    title = str(num_epochs) + " epochs with "+ loss_function_str + " and lr=" + str(learning_rate)
    subtitle = "nTrainingBatches = " + str(num_training_batches)+ " TrainBatchSize = " + str(train_batch_size)+ " TestBatchSize = "+ str(test_batch_size)+ " nTestingBatches = " + str(num_test_batches)


    train_loss_list = json.loads(data[9])
    test_loss_list = json.loads(data[10])



    significances = []

    for i in range(len(tp)):
        significances.append(tp[i]/math.sqrt(float(tp[i]) + float(fp[i]) + 0.00000001))


    plt.figure(figsize=(20,20))
    font1 = {'size':10}
    font2 = {'size':40}

    plt.subplot(4, 2, 1)
    plt.title("Training", fontdict = font1)
    plt.ylabel("Training Loss (per batch)", fontdict = font1)
    plt.plot(train_loss_list)


    plt.subplot(4, 2, 2)
    plt.title("Testing", fontdict = font1)
    plt.xlabel("Epoch number", fontdict = font1)
    plt.ylabel("Test Loss", fontdict = font1)
    plt.plot(test_loss_list)


    plt.subplot(4, 2, 3)
    plt.title("Significance", fontdict = font1)
    plt.xlabel("Epoch number", fontdict = font1)
    plt.ylabel("Testing Significance", fontdict = font1)
    plt.plot(significances)


    plt.subplot(4, 2, 4)
    plt.title("Accuracy", fontdict = font1)
    plt.xlabel("Epoch number", fontdict = font1)
    plt.ylabel("percent accuracy", fontdict = font1)
    plt.plot(accuracy)


    plt.subplot(4, 2, 5)
    plt.title("Fp verses Tp", fontdict = font1)
    plt.xlabel("Epoch number", fontdict = font1)
    plt.ylabel("Fp and Tp count", fontdict = font1)
    plt.plot(tp)
    plt.plot(fp)

    plt.subplot(4,2,6)
    plt.title("ROC curve", fontdict = font1)

    plt.xlabel("False Positive Rate", fontdict = font1)
    plt.ylabel("True Positive Rate", fontdict = font1)

    cutoffs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1]
    if (test_data == None) or (test_target_and_weights == None):
        test_data, test_target_and_weights = open_test_data(v.num_testing_batches)

    test_target, test_weights = split_weights_from_target(test_target_and_weights)

    tp_roc,fp_roc = calculate_roc_curve_points(cutoffs,network,loss_function_id,test_data,test_target, test_weights)

    plt.plot(fp_roc,tp_roc)


    plt.subplot(4,2,7)
    plt.title("Asimov Estimate of Significance", fontdict = font1)

    plt.xlabel("training batch", fontdict = font1)
    plt.ylabel("Asimov estimate of significance", fontdict = font1)
    asimov_loss_list = []
    for i in range(len(tp)):
        asimov_loss_list.append(asimov_from_tp_fp(tp[i],fp[i],0.5))
    plt.plot(asimov_loss_list)



    plt.subplot(4,2,8)

    table_data = [
    ["Final Significance", round(significances[-1],4)],
    ["Final Training Loss", round(train_loss_list[-1],4)],
    ["Final Test Loss", round(test_loss_list[-1],4)]]
    #["Final accuracy",correct[-1]]]

    table = plt.table(cellText=table_data, loc='center')


    plt.suptitle(str(title),fontdict = font2)
    with open(plot_path,"w") as plot_file:
        plt.savefig(plot_path)
