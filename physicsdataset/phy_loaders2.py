import csv

from tqdm import tqdm
import numpy as np
import torch

train_batch_size = 10
num_variables = 30

def open_training_data(number_of_batches, pickle=True):

    if number_of_batches * train_batch_size > 200000:
        print("Warning! requested too much training data, only 200000 records available, {} requested".format(number_of_batches * train_batch_size))
        return "warning"

    training_data = torch.zeros((number_of_batches,train_batch_size,num_variables)) #variables by number of events in one batch by number of batches
    training_target = torch.zeros((number_of_batches,train_batch_size,2))

    with open("training.csv") as training_data_file:
        training_data_file.seek(0)
        trainreader = csv.reader(training_data_file)
        next(trainreader)

        for batch in tqdm(range(number_of_batches), colour = "black",desc= "Loading Training Data"):

            for event in range(train_batch_size):
                line = next(trainreader)

                for variable in range(1,num_variables+1):
                    training_data[batch,event,variable-1] = float(line[variable])/1
                print(training_data[batch,event])
                print(len(training_data[batch,event]))
                if line[32]=='s':
                    training_target[batch][event][0] = 1
                else:
                    training_target[batch][event][0] = 0
                training_target[batch][event][1] = float(line[31])
            print(training_target)
    if pickle:
        print("pickling training data...")
        data_path = "non_normalized_loaded_data/train_data_nb_" + str(number_of_batches) + "_bs_" + str(train_batch_size) + ".pt"
        target_path = "non_normalized_loaded_data/train_target_nb_" + str(number_of_batches) + "_bs_" + str(train_batch_size) + ".pt"

        torch.save(training_data,data_path)

        torch.save(training_target,target_path)
        return training_data,training_target
    return training_data,training_target

open_training_data(1)
