import numpy as np
from model import Model

class SVDPlusPlusSimplified(Model):
    # stochastic gradient descent with weighted lambda regularisation
    name = 'SVD++ Simplified'

    def __init__(self, train_X, test_X, threshold=.5, n_epochs=100, k=20, gamma=(.005, .95), lmbda=.050):
        super(SVDPlusPlusSimplified, self).__init__(train_X, test_X, threshold)

        self.n_epochs = n_epochs  # Number of epochs
        self.k = k  # Dimensionality of the latent feature space
        self.gamma = gamma # Learning rate
        self.lmbda = lmbda # Regularisation weight

    def get_title(self):
        title = '{0} Learning Curve (k={1},gamma={2},lmbda={3}), correctness threshold={4}'.format(self.name, self.k, self.gamma, self.lmbda, self.threshold)
        return title

    def initialize_weights(self):
        self.I = self.train_X.copy()
        self.I[self.I > 0] = 1
        self.I[self.I == 0] = 0

        self.N = np.sum(self.train_X, axis=1)
        self.N[self.N == 0] = 1 #debatable, either need this or put ifs back

        self.I2 = self.test_X.copy()
        self.I2[self.I2 > 0] = 1
        self.I2[self.I2 == 0] = 0

        self.g = self.gamma[0]

        self.P = 1 * np.random.rand(self.m,self.k) # Latent user feature matrix
        self.Q = 1 * np.random.rand(self.n,self.k) # Latent movie feature matrix
        self.Y = .1 * np.random.rand(self.n,self.k) # Implicit feedback feature matrix

    def SGD_step(self, u, i):
        e = self.train_X[u, i] - self.predict_single(u, i)  # Calculate error for gradient
        # time1 += time.time() - cur
        # cur = time.time()
        # if N[u]:
        self.Q[i] += self.g * (e * (self.P[u] + self.N[u]**-.5 * np.dot(self.Y.T, self.I[u])) - self.lmbda * self.Q[i])  # Update latent movie feature matrix
        # else:
            # Q[i] += gamma * (e * (P[u] + np.dot(Y.T, I[u])) - lmbda * Q[i])  # Update latent movie feature matrix
        # time2 += time.time() - cur
        # cur = time.time()
        self.P[u] += self.g * (e * self.Q[i] - self.lmbda * self.P[u]) # Update latent user feature matrix
        # time3 += time.time() - cur
        # cur = time.time()
        Nu = self.train_X[u].nonzero()[0] # this literally optimized runtime by 1000x
        self.Y[Nu] *= (1 - self.g * self.lmbda)
        # if N[u]:
        Yd = self.g * (e * self.N[u]**-.5 * self.Q[i])
        # else:
            # Yd = gamma * (e * Q[i])
        np.add(self.Y[Nu], Yd)

    def get_train_test_rmse(self):
        train_rmse = self.rmse_train() # Calculate root mean squared error from train dataset
        test_rmse = self.rmse_test() # Calculate root mean squared error from test dataset
        return train_rmse, test_rmse

    def get_train_test_percent_error(self):
        train_error_percent = self.percent_error_train()
        test_error_percent = self.percent_error_test()
        return train_error_percent, test_error_percent

    def update_learning_rate(self):
        self.g *= self.gamma[1]

    def predict_all(self):
        pred = np.zeros(self.train_X.shape)
        for u in range(self.m):
            for i in range(self.n):
                pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_train(mu,Bu,Bi,P,Q,Y,I,N):
        pred = np.zeros(self.train_X.shape)
        for u, i in self.user_item_pairs:
            pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_test(mu,Bu,Bi,P,Q,Y,I,N):
        pred = np.zeros(self.train_X.shape)
        for u, i in self.user_item_pairs_test:
            pred[u, i] = self.predict_single(u, i)
        return pred

    def predict_single(self, u, i):
        # if N:
        return np.dot(self.Q[i], self.P[u] + self.N[u]**-.5 * np.dot(self.Y.T, self.I[u]))
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



