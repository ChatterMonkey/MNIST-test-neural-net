

class variable_list:
    def __init__(self):
        self.signal = 4
        self.background = 7
        self.train_batch_size = 4000
        self.test_batch_size = 10000
        self.mnist_test_size = 10000
        self.mnist_train_size = 60000
        self.momentum = 0.5
        self.log_interval = 5 #how often to log while training
        self.random_seed = 7

        self.n_epochs = None
        self.learning_rate = None

    def set_lr(self,lr):
        self.learning_rate = lr
    def set_n_epochs(self,n_epochs):
        self.n_epochs = n_epochs

variables = variable_list()
