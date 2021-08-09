class variable_list:
    def __init__(self):

        self.train_batch_size = None
        self.test_batch_size = None
        self.n_epochs = None
        self.learning_rate = None

    def set_lr(self,lr):
        self.learning_rate = lr

    def set_train_batch_size(self,train_batch_size):
        self.train_batch_size = train_batch_size

    def set_test_batch_size(self,test_batch_size):
        self.test_batch_size = test_batch_size

    def set_n_epochs(self,n_epochs):
        self.n_epochs = n_epochs

variables = variable_list()
