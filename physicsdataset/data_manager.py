import sqlite3 as sq
import json
from physicsdataset.phy_variables import variables as v
from physicsdataset.phys_roc_maker import calculate_roc_curve_points
from physicsdataset.phy_net import Net
import matplotlib.pyplot as plt
import math

connection = sq.connect("data.sql")
cur = connection.cursor()
e = cur.execute

e('DROP TABLE data')


def add_data(network_state_dict, training_loss,testing_loss,accuracy,tp_list,fp_list):
    network_string = json.dumps(network_state_dict)

    trainlj = json.dumps(training_loss)
    testlj = json.dumps(testing_loss)
    accj = json.dumps(accuracy)
    tp = json.dumps(tp_list)
    fp = json.dumps(fp_list)

    e('CREATE TABLE IF NOT EXISTS data(id integer primary key autoincrement, network, train_batch_size, test_batch_size, num_training_batches, num_testing_batches, loss_function_id, learning_rate, num_epochs,training_loss,testing_loss, accuracy,tp,fp)')
    e('INSERT INTO data VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(None, network_string,v.train_batch_size,v.test_batch_size,v.num_training_batches,v.num_testing_batches,v.loss_function_id,v.learning_rate,v.num_epochs, trainlj,testlj,accj,tp,fp))
    print(e('Select * from data').fetchall())


def visulize(experiment_id = 0, plot_last = False):
    print("visulizing")
    print(experiment_id)
    cur = connection.cursor()
    if plot_last:
        data = cur.execute('SELECT * FROM data ORDER BY id DESC LIMIT 1').fetchall()
        print(data)
    else:
        data = cur.execute('SELECT * FROM data WHERE id = ?',(experiment_id,)).fetchall()
        print(data)
    data = data[0]

    print("id , train_batch_size, test_batch_size, num_training_batches, num_testing_batches, loss_function_id, learning_rate, num_epochs,training_loss,testing_loss, accuracy,tp,fp")

    network = Net()
    network.load_state_dict(data[2])
    loss_function_str = v.loss_function_tuple[data[6]][0]
    learning_rate = data[7]
    num_epochs = data[8]
    num_training_batches = data[4]
    train_batch_size = data[2]
    num_test_batches = data[5]
    test_batch_size = data[3]

    accuracy = json.loads(data[11])

    tp = json.loads(data[12])
    print(tp)
    fp = json.loads(data[13])



    title = str(num_epochs) + " epochs with "+ loss_function_str + " and lr=" + str(learning_rate)
    subtitle = "nTrainingBatches = " + str(num_training_batches)+ " TrainBatchSize = " + str(train_batch_size)+ " TestBatchSize = "+ str(test_batch_size)+ " nTestingBatches = " + str(num_test_batches)
    print(title)
    print(subtitle)

    train_loss_list = json.loads(data[9])
    test_loss_list = json.loads(data[10])
    print(test_loss_list)


    significances = []
    if loss_function_id == 1:
        for i in range(len(test_loss_list)):
            significances.append(math.sqrt(-1 * test_loss_list[i][0]))
    else:
        print("using seperate sig evaluation")
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

    cutoffs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    true_positive_rates, false_positive_rates  = calculate_roc_curve_points(cutoffs,network,loss_function_id,30)

    #plt.plot(false_positive_rates,true_positive_rates)

    #plt.subplot(4,2,7)

    table_data = [
    ["Final Significance", round(significances[-1],4)],
    ["Final Training Loss", round(train_loss_list[-1],4)],
    ["Final Test Loss", round(test_loss_list[-1],4)]]
    #["Final accuracy",correct[-1]]]

    table = plt.table(cellText=table_data, loc='center')


    plt.suptitle(str(title),fontdict = font2)
    plt.savefig('../physics_graphs/test_graph.png')





loss_function_id = 2
num_epochs = 2
learning_rate = 0.001

num_training_batches = 1
num_testing_batches = 4
train_batch_size = 2
test_batch_size = 3



v.set_params(train_batch_size,test_batch_size,num_training_batches,num_testing_batches,loss_function_id,learning_rate,num_epochs)

training_loss = [1,2,3]
testing_loss = [5,6,7]
accuracy = [0,6,7]

tp = [9,6,7]
fp = [10,3,4]


add_data(training_loss,testing_loss, accuracy,tp,fp)

print(e('SELECT * FROM data WHERE id = 2').fetchall())
print(e('''SELECT * FROM data ORDER BY id DESC LIMIT 1''').fetchall())

visulize(experiment_id=1)
