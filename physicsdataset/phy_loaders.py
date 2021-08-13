import csv
from physicsdataset.phy_variables import variables

def open_training_data(number_of_batches):

    if number_of_batches * variables.train_batch_size > 200000:
        print("Warning! requested too much training data, only 200000 records available, {} requested".format(number_of_batches * variables.train_batch_size))
        return "warning"

    training_data_lists = []
    training_target_lists = []
    with open("training.csv") as training_data:
        training_data.seek(0)
        trainreader = csv.reader(training_data)
        next(trainreader)

        for j in range(number_of_batches):
            train_data_list = []
            train_target_list = []
            for i in range(variables.train_batch_size):
                line = next(trainreader)
                train_data_list.append(line[1:31])
                if line[32] == 's':
                    #print("signal")
                    train_target_list.append(1)
                else:
                    #print("bkground")
                    train_target_list.append(0)
                #print(train_target_list)
            training_data_lists.append(train_data_list)
            training_target_lists.append(train_target_list)
            #print( training_target_lists)

        return training_data_lists,training_target_lists



def open_test_data(number_of_batches):

    if number_of_batches * variables.test_batch_size > 50000:
        print("Warning! requested too much testing data, only 50000 records available, {} requested".format(number_of_batches * variables.test_batch_size))

    test_data_lists = []
    test_target_lists = []

    with open("training.csv") as testing_data:
        testreader = csv.reader(testing_data)
        for i in range(200001):
        #for i in range(1):
            next(testreader)

        for j in range(number_of_batches):
            test_data_list = []
            test_target_list = []
            for i in range(variables.test_batch_size):
                line = next(testreader)
                test_data_list.append(line[1:31] )
                if line[32] == 's':
                    test_target_list.append(1)
                else:
                    test_target_list.append(0)
            test_data_lists.append(test_data_list)
            test_target_lists.append(test_target_list)

        return test_data_lists,test_target_lists


