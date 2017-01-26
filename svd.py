import numpy as np
from model import Model

class SVD(Model):
    # stochastic gradient descent with weighted lambda regularisation
    name = 'SVD'

    def __init__(self, train_X, test_X, threshold=.5, n_epochs=100, k=20, gamma=.005, lmbda=.1):
        super(SVD, self).__init__(train_X, test_X, threshold)

        self.n_epochs = n_epochs  # Number of epochs
        self.k = k  # Dimensionality of the latent feature space
        self.gamma = gamma  # Learning rate
        self.lmbda = lmbda # Regularisation weight

    def get_title(self):
        title = '{0} Learning Curve (k={1},gamma={2},lmbda={3}), correctness threshold={4}'.format(self.name, self.k, self.gamma, self.lmbda, self.threshold)
        return title

    def initialize_weights(self):
        #Only consider non-zero matrix 
        users,items = self.train_X.nonzero()
        users_test,items_test = self.test_X.nonzero()
        self.user_item_pairs = zip(users,items)
        self.user_item_pairs_test = zip(users_test,items_test)

        self.I = self.train_X.copy()
        self.I[self.I > 0] = 1
        self.I[self.I == 0] = 0

        self.I2 = self.test_X.copy()
        self.I2[self.I2 > 0] = 1
        self.I2[self.I2 == 0] = 0

        self.g = self.gamma  # Learning rate

        self.P = 1 * np.random.rand(self.m,self.k) # Latent user feature matrix
        self.Q = 1 * np.random.rand(self.n,self.k) # Latent movie feature matrix

    def SGD_step(self, u, i):
        e = self.train_X[u, i] - self.predict_single(u,i)  # Calculate error for gradient
        self.P[u] += self.g * ( e * self.Q[i] - self.lmbda * self.P[u]) # Update latent user feature matrix
        self.Q[i] += self.g * ( e * self.P[u] - self.lmbda * self.Q[i])  # Update latent movie feature matrix

    def get_train_test_rmse(self):
        train_rmse = self.rmse(self.I,self.train_X) # Calculate root mean squared error from train dataset
        test_rmse = self.rmse(self.I2,self.test_X) # Calculate root mean squared error from test dataset
        return train_rmse, test_rmse

    def get_train_test_percent_error(self):
        return 0, 0

    # def update_learning_rate(self):
    #     pass

    def predict_all(self):
        return np.dot(self.P,self.Q.T)

    def predict_single(self, u, i):
        return np.dot(self.P[u],self.Q[i])

    def rmse(self, I, R):
        return np.sqrt(np.sum((I * (R - self.predict_all()))**2)/len(R[R > 0]))




