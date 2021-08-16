class variable_list:
    def __init__(self):

        self.num_variables = 30
        self.normalization_constant = 300

        self.train_batch_size = None
        self.test_batch_size = None

        self.num_training_batches = None
        self.num_testing_batches = None


        self.loss_function_id = None
        self.learning_rate = None
        self.num_epochs = None

        self.expectedSignal = 1
        self.expectedBackground = 1
        self.loss_function_tuple = (("mean squared error","mse"),("significance loss","sl"),("binery cross entropy","bce"))

    def set_params(self,train_batch_size,test_batch_size,num_training_batches,num_testing_batches,loss_function_id,learning_rate,num_epochs):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.num_training_batches = num_training_batches
        self.num_testing_batches = num_testing_batches

        self.loss_function_id = loss_function_id
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

variables = variable_list()
