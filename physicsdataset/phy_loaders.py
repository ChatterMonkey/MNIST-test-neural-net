import csv
from physicsdataset.phy_variables import variables
from tqdm import tqdm
import numpy as np
import torch



def open_training_data(number_of_batches):

    if number_of_batches * variables.train_batch_size > 200000:
        print("Warning! requested too much training data, only 200000 records available, {} requested".format(number_of_batches * variables.train_batch_size))
        return "warning"

    training_data = torch.zeros((number_of_batches,variables.train_batch_size,variables.num_variables)) #variables by number of events in one batch by number of batches
    training_target = torch.zeros((number_of_batches,variables.train_batch_size,1))

    with open("training.csv") as training_data_file:
        training_data_file.seek(0)
        trainreader = csv.reader(training_data_file)
        next(trainreader)

        for batch in tqdm(range(number_of_batches), colour = "black",desc= "Loading Training Data"):

            for event in range(variables.train_batch_size):
                line = next(trainreader)

                for variable in range(1,variables.num_variables+1):
                    training_data[batch,event,variable-1] = float(line[variable])/variables.normalization_constant

                if line[32]=='s':
                    training_target[batch][event][0] = 1
                else:
                    training_target[batch][event][0] = 0

        return training_data,training_target



def open_test_data(number_of_batches):

    if number_of_batches * variables.test_batch_size > 50000:
        print("Warning! requested too much testing data, only 50000 records available, {} requested".format(number_of_batches * variables.test_batch_size))

    testing_data = torch.zeros((number_of_batches,variables.train_batch_size,variables.num_variables)) #variables by number of events in one batch by number of batches
    testing_target = torch.zeros((number_of_batches,variables.train_batch_size,1))

    with open("training.csv") as testing_data_file:
        testreader = csv.reader(testing_data_file)
        for i in range(200001):
            next(testreader)

        for batch in tqdm(range(number_of_batches), colour="black",desc= "Loading Testing Data"):
            for event in range(variables.test_batch_size):
                line = next(testreader)
                for variable in range(1,variables.num_variables+1):
                    testing_data[batch][event][variable-1] = float(line[variable])/variables.normalization_constant
                if line[32]=='s':
                    testing_target[batch][event][0] = 1
                else:
                    testing_target[batch][event][0] = 0
        return testing_data,testing_target


