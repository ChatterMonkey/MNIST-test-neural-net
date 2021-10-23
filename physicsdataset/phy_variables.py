class variable_list:
    def __init__(self):

        self.num_variables = 30
        #self.normalization_constant =300
        self.normalization_constant = 1

        self.train_batch_size = None
        self.test_batch_size = None

        self.num_training_batches = None
        self.num_testing_batches = None


        self.loss_function_id = None
        self.learning_rate = None
        self.num_epochs = None

        self.expectedSignal = 0.34
        self.expectedBackground = 0.657
        self.systematic = None
        self.loss_function_tuple = (("mean squared error","mse"),("significance loss","sl"),("binery cross entropy","bce"),("asimov estimate","ae"),("inverted significance loss","isl"))

    def set_params(self,train_batch_size,test_batch_size,num_training_batches,num_testing_batches,loss_function_id,learning_rate,num_epochs,systematic):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.num_training_batches = num_training_batches
        self.num_testing_batches = num_testing_batches

        self.loss_function_id = loss_function_id
        self.learning_rate = learning_rate
        self.systematic = systematic
        self.num_epochs = num_epochs

    def restore_epochs(self, epochs):
        self.num_epochs = epochs
variables = variable_list()
