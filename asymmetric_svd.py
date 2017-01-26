import numpy as np
from model import Model
import time

class AsymmetricSVD(Model):
    # stochastic gradient descent with weighted lambda regularisation
    name = 'Asymmetric SVD'
    path = 'asym_svd'

    def __init__(self, train_X, test_X, threshold=.5, in_folder=None, out_folder=None, n_epochs=100, k=100, gamma=(.007,.007,.007,.95), lmbda=(.040,.040,.040)):
        super(AsymmetricSVD, self).__init__(train_X, test_X, threshold, in_folder, out_folder)

        self.n_epochs = n_epochs  # Number of epochs
        self.k = k  # Dimensionality of the latent feature space
        self.gamma = gamma # Learning rate
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

        self.N = np.sum(self.train_X, axis=1)
        self.N[self.N == 0] = 1 #debatable, either need this or put ifs back

        self.I2 = self.test_X.copy()
        self.I2[self.I2 > 0] = 1
        self.I2[self.I2 == 0] = 0

        self.g = [self.gamma[0], self.gamma[1], self.gamma[2]]  # Learning rate

        self.mu = np.sum(self.train_X) / len(self.user_item_pairs) # total mean of all scores
        # try initializing with dif between user/anime mean and total mean
        self.bu = 25
        self.bi = 25
        self.offset = self.I * (self.train_X - self.mu)
        self.Bu = np.sum(self.offset, axis=1) / (np.sum(self.I, axis=1) + self.bu)
        self.offset = self.I * (self.offset - self.Bu[:, None])
        self.Bi = np.sum(self.offset, axis=0) / (np.sum(self.I, axis=0) + self.bi)
        self.offset = self.I * (self.offset - self.Bi)
        self.Bu = 1 * np.random.rand(self.m) # user biases
        self.Bi = 1 * np.random.rand(self.n) # anime biases

        self.Q = .1 * np.random.rand(self.n,self.k) # Latent movie feature matrix
        self.W = .01 * np.random.rand(self.n,self.k) # Baseline offset similarity feature matrix
        self.C = .01 * np.random.rand(self.n,self.k) # Implicit feedback feature matrix

    def SGD_step(self, u, i):
        e = self.train_X[u, i] - self.predict_single(u, i)  # Calculate error for gradient
        # time0 += time.time() - cur
        # cur = time.time()
        self.Bu[u] += self.g[0] * (e - self.lmbda[0] * self.Bu[u])
        self.Bu[i] += self.g[0] * (e - self.lmbda[0] * self.Bu[i])
        # time1 += time.time() - cur
        # cur = time.time()
        # if N[u]:
        self.Q[i] += self.g[1] * (e * (self.N[u]**-.5 * (np.dot(self.W.T, self.offset[u]) + np.dot(self.C.T, self.I[u]))) - self.lmbda[1] * self.Q[i])
        # else:
            # Q[i] += gamma * (e * (P[u] + np.dot(Y.T, I[u])) - lmbda * Q[i])  # Update latent movie feature matrix
        # time2 += time.time() - cur
        # cur = time.time()
        Nu = self.train_X[u].nonzero()[0] # this literally optimized runtime by 1000x
        self.W[Nu] *= (1 - self.g[2] * self.lmbda[2])
        # if N[u]:
        Wd = self.g[1] * (e * self.N[u]**-.5 * self.Q[i])
        Wd = self.offset[u][Nu][:, None] * np.tile(Wd, (len(Nu), 1))
        # self.W[Nu] = np.add(self.W[Nu], Wd)
        self.W[Nu] += Wd
        # time3 += time.time() - cur
        # cur = time.time()
        # Nu = self.train_X[u].nonzero()[0] # this literally optimized runtime by 1000x
        self.C[Nu] *= (1 - self.g[2] * self.lmbda[2])
        # if N[u]:
        Cd = self.g[1] * (e * self.N[u]**-.5 * self.Q[i])
        # else:
            # Yd = gamma * (e * Q[i])
        self.C[Nu] = np.add(self.C[Nu], Cd)

    def get_train_test_both_error(self):
        train_rmse, train_error_percent = self.both_error_train()
        test_rmse, test_error_percent = self.both_error_test()
        return train_rmse, test_rmse, train_error_percent, test_error_percent

    def get_train_test_rmse(self):
        train_rmse = self.rmse_train() # Calculate root mean squared error from train dataset
        test_rmse = self.rmse_test() # Calculate root mean squared error from test dataset
        return train_rmse, test_rmse

    def get_train_test_percent_error(self):
        train_error_percent = self.percent_error_train()
        test_error_percent = self.percent_error_test()
        return train_error_percent, test_error_percent

    def update_learning_rate(self):
        self.g[0] *= self.gamma[3]
        self.g[1] *= self.gamma[3]
        self.g[2] *= self.gamma[3]

    def predict_all(self):
        pred = np.zeros(self.train_X.shape)
        for u in range(self.m):
            for i in range(self.n):
                pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_train(mu,Bu,Bi,P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u, i in self.user_item_pairs:
            pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_test(mu,Bu,Bi,P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u, i in self.user_item_pairs_test:
            pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_single(self, u, i):
        # if N:
        return self.mu + self.Bu[u] + self.Bi[i] + np.dot(self.Q[i], self.N[u]**-.5 * (np.dot(self.W.T, self.offset[u]) + np.dot(self.C.T, self.I[u])))
        # return np.dot(Q, P)

    def rmse(self, R, ui_pairs):
        sq_err = 0
        for u, i in ui_pairs:
            sq_err += (R[u, i] - self.predict_single(u, i))**2
        return np.sqrt(sq_err/len(ui_pairs))

    def rmse_train(self):
        return self.rmse(self.train_X, self.user_item_pairs)

    def rmse_test(self):
        return self.rmse(self.test_X, self.user_item_pairs_test)

    def percent_error(self, R, ui_pairs):
        wrong = 0
        for u, i in ui_pairs:
            if abs(R[u, i] - self.predict_single(u, i)) > self.threshold:
                wrong += 1
        return float(wrong)/len(ui_pairs)

    def percent_error_train(self):
        return self.percent_error(self.train_X, self.user_item_pairs)

    def percent_error_test(self):
        return self.percent_error(self.test_X, self.user_item_pairs_test)

    def both_error(self, R, ui_pairs):
        sq_err = 0
        wrong = 0
        for u, i in ui_pairs:
            pred = self.predict_single(u, i)
            sq_err += (R[u, i] - pred)**2
            if abs(R[u, i] - pred) > self.threshold:
                wrong += 1
        return np.sqrt(sq_err/len(ui_pairs)), float(wrong)/len(ui_pairs)

    def both_error_train(self):
        return self.both_error(self.train_X, self.user_item_pairs)

    def both_error_test(self):
        return self.both_error(self.test_X, self.user_item_pairs_test)

