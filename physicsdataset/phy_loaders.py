import csv

def opendata(number_of_rows):
    training_data_list = []
    training_targets = []

    with open("training.csv") as training_data:

        trainreader = csv.reader(training_data)
        next(trainreader)
        for i in range(1,number_of_rows):
            line = next(trainreader)
            training_data_list.append(line[1:31] )
            if line[32] == 's':
                training_targets.append(1)
            else:
                training_targets.append(0)

        return training_data_list,training_targets



