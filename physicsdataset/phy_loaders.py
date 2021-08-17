import csv
from physicsdataset.phy_variables import variables
from tqdm import tqdm
import numpy as np



def open_training_data(number_of_batches):

    if number_of_batches * variables.train_batch_size > 200000:
        print("Warning! requested too much training data, only 200000 records available, {} requested".format(number_of_batches * variables.train_batch_size))
        return "warning"

    training_data = np.zeros((variables.train_batch_size,variables.num_variables,number_of_batches), dtype=float) #variables by number of events in one batch by number of batches
    training_target = np.zeros((variables.train_batch_size,1, number_of_batches))

    with open("training.csv") as training_data:
        training_data.seek(0)
        trainreader = csv.reader(training_data)
        next(trainreader)

        for batch in tqdm(range(number_of_batches), colour = "black",desc= "Loading Training Data"):

            for event in range(variables.train_batch_size):
                line = next(trainreader)
                for variable in range(variables.num_variables):
                    training_data[event][variable][batch] = line[variable]

                if line[32]=='s':
                    training_target[event][0][batch] = 1
                else:
                    training_target[event][0][batch] = 1


        return training_data,training_target



def open_test_data(number_of_batches):
    number_of_signals = 0
    number_of_background_events = 0
    if number_of_batches * variables.test_batch_size > 50000:
        print("Warning! requested too much testing data, only 50000 records available, {} requested".format(number_of_batches * variables.test_batch_size))

    test_data_lists = []
    test_target_lists = []

    with open("training.csv") as testing_data:
        testreader = csv.reader(testing_data)
        for i in range(200001):
            next(testreader)

        for j in tqdm(range(number_of_batches), colour="black",desc= "Loading Testing Data"):
            test_data_list = []
            test_target_list = []
            for i in range(variables.test_batch_size):
                line = next(testreader)
                test_data_list.append(line[1:31] )
                if line[32] == 's':
                    number_of_signals += 1
                    test_target_list.append(1)
                else:
                    number_of_background_events += 1
                    test_target_list.append(0)
            test_data_lists.append(test_data_list)
            test_target_lists.append(test_target_list)
        print("")
        print("s/b is {}/{} = {}%".format(number_of_signals,number_of_background_events,number_of_signals/number_of_background_events*100))

        return test_data_lists,test_target_lists


