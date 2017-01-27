import json
import numpy as np
from model import Model
import os
import time

class SVDPlusPlus(Model):
    # stochastic gradient descent with weighted lambda regularisation
    name = 'SVD++'
    path = 'svdpp'

    def __init__(self, train_X, test_X, threshold=.5, in_folder=None, out_folder=None, n_epochs=100, k=20, gamma=(.007,.005,.95), lmbda=(.050,.050)):
        super(SVDPlusPlus, self).__init__(train_X, test_X, threshold, in_folder, out_folder)

        self.n_epochs = n_epochs  # Number of epochs
        self.k = k  # Dimensionality of the latent feature space
        self.gamma = gamma # Learning rate
        self.lmbda = lmbda # Regularisation weight

        np.random.seed(42)

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

        self.mu = np.sum(self.train_X) / len(self.user_item_pairs) # total mean of all scores

        if self.in_folder != None:
            self.load_weights()
        else:
            self.g = [self.gamma[0], self.gamma[1]]  # Learning rate

            # try initializing with dif between user/anime mean and total mean
            self.Bu = 1 * np.random.rand(self.m) # user biases
            self.Bi = 1 * np.random.rand(self.n) # anime biases

            self.P = .1 * np.random.rand(self.m,self.k) # Latent user feature matrix
            self.Q = .1 * np.random.rand(self.n,self.k) # Latent movie feature matrix
            self.Y = .1 * np.random.rand(self.n,self.k) # Implicit feedback feature matrix

    def save_weights(self):
        if not os.path.exists("weights/{0}/{1}".format(self.path, self.out_folder)):
            os.makedirs("weights/{0}/{1}".format(self.path, self.out_folder))
        model_metadata = {
            'g': self.g,
        }
        with open("weights/{0}/{1}/{2}.json".format(self.path, self.out_folder, 'model_metadata'), 'w') as f:
            json.dump(model_metadata, f)
        np.save("weights/{0}/{1}/{2}.npy".format(self.path, self.out_folder, 'Bu'), self.Bu)
        np.save("weights/{0}/{1}/{2}.npy".format(self.path, self.out_folder, 'Bi'), self.Bi)
        np.save("weights/{0}/{1}/{2}.npy".format(self.path, self.out_folder, 'P'), self.P)
        np.save("weights/{0}/{1}/{2}.npy".format(self.path, self.out_folder, 'Q'), self.Q)
        np.save("weights/{0}/{1}/{2}.npy".format(self.path, self.out_folder, 'Y'), self.Y)

    def load_weights(self):
        with open("weights/{0}/{1}/{2}.json".format(self.path, self.in_folder, 'model_metadata'), 'r') as f:
            model_metadata = json.load(f)
        self.g = model_metadata['g']
        self.Bu = np.load("weights/{0}/{1}/{2}.npy".format(self.path, self.in_folder, 'Bu'))
        self.Bi = np.load("weights/{0}/{1}/{2}.npy".format(self.path, self.in_folder, 'Bi'))
        self.P = np.load("weights/{0}/{1}/{2}.npy".format(self.path, self.in_folder, 'P'))
        self.Q = np.load("weights/{0}/{1}/{2}.npy".format(self.path, self.in_folder, 'Q'))
        self.Y = np.load("weights/{0}/{1}/{2}.npy".format(self.path, self.in_folder, 'Y'))

    def SGD_step(self, u, i):
        cur = time.time()
        e = self.train_X[u, i] - self.predict_single(u, i)  # Calculate error for gradient
        self.time0 += time.time() - cur
        cur = time.time()
        self.Bu[u] += self.g[0] * (e - self.lmbda[0] * self.Bu[u])
        self.Bu[i] += self.g[0] * (e - self.lmbda[0] * self.Bu[i])
        self.time1 += time.time() - cur
        cur = time.time()
        # if N[u]:
        self.Q[i] += self.g[1] * (e * (self.P[u] + self.N[u]**-.5 * np.dot(self.Y.T, self.I[u])) - self.lmbda[1] * self.Q[i])  # Update latent movie feature matrix
        # else:
            # Q[i] += gamma * (e * (P[u] + np.dot(Y.T, I[u])) - lmbda * Q[i])  # Update latent movie feature matrix
        self.time2 += time.time() - cur
        cur = time.time()
        self.P[u] += self.g[1] * (e * self.Q[i] - self.lmbda[1] * self.P[u]) # Update latent user feature matrix
        self.time3 += time.time() - cur
        cur = time.time()
        Nu = self.train_X[u].nonzero()[0] # this literally optimized runtime by 1000x
        self.Y[Nu] *= (1 - self.g[1] * self.lmbda[1])
        # if N[u]:
        Yd = self.g[1] * (e * self.N[u]**-.5 * self.Q[i])
        # else:
            # Yd = gamma * (e * Q[i])
        # self.Y[Nu] = np.add(self.Y[Nu], Yd)
        Yd = np.tile(Yd, (len(Nu), 1))
        self.Y[Nu] += Yd
        self.time4 += time.time() - cur

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
        self.g[0] *= self.gamma[2]
        self.g[1] *= self.gamma[2]

    def predict_all(self):
        pred = np.zeros(self.train_X.shape)
        for u in range(self.m):
            for i in range(self.n):
                pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_train(self):
        pred = np.zeros(train_X.shape)
        for u, i in self.user_item_pairs:
            pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_test(self):
        pred = np.zeros(train_X.shape)
        for u, i in self.user_item_pairs_test:
            pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_single(self, u, i):
        # if N:
        return self.mu + self.Bu[u] + self.Bi[i] + np.dot(self.Q[i], self.P[u] + self.N[u]**-.5 * np.dot(self.Y.T, self.I[u]))
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



