class variable_list:
    def __init__(self):

        self.num_variables = 30
        self.normalization_constant = 300

        self.train_batch_size = None
        self.test_batch_size = None
        self.n_epochs = None
        self.learning_rate = None
        self.num_epochs = None

        self.expectedSignal = 1
        self.expectedBackground = 1

    def set_lr(self,lr):
        self.learning_rate = lr

    def set_epochs(self,num_epochs):
        self.num_epochs = num_epochs

    def set_train_batch_size(self,train_batch_size):
        self.train_batch_size = train_batch_size

    def set_test_batch_size(self,test_batch_size):
        self.test_batch_size = test_batch_size

    def set_n_epochs(self,n_epochs):
        self.n_epochs = n_epochs

variables = variable_list()
