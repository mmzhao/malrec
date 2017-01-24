import matplotlib.pyplot as plt
import numpy as np
import time

class Model(object):
    # Base class for all svd with sgd based models
    name = 'Generic'

    def __init__(self, train_X, test_X, threshold):
        self.train_X = train_X
        self.test_X = test_X
        self.threshold = threshold

        self.m, self.n = train_X.shape  # Number of users and items

    def get_title(self):
        pass

    def initialize_weights(self):
        pass

    def SGD_step(self, u, i):
        pass

    def get_train_test_rmse(self):
        pass

    def get_train_test_percent_error(self):
        pass

    def update_learning_rate(self):
        pass

    # def rmse(self, R):
    #     pass

    def predict_all(self):
        pass

    def predict_single(self, u, i):
        pass

    def train(self, id2anime):
        print "Training {0} model".format(self.name)

        #Only consider non-zero matrix 
        users,items = self.train_X.nonzero()
        users_test,items_test = self.test_X.nonzero()
        self.user_item_pairs = zip(users,items)
        self.user_item_pairs_test = zip(users_test,items_test)

        self.initialize_weights()

        train_errors = []
        test_errors = []
        train_error_percents = []
        test_error_percents = []

        title = self.get_title()
        print title

        try:
            start_time = time.time()

            train_rmse, test_rmse = self.get_train_test_rmse()
            train_error_percent, test_error_percent = self.get_train_test_percent_error()
            
            print "[Epoch %d/%d, time %f]\ntrain error: %f, test error: %f\ntrain error percent: %f, test error percent: %f" \
            %(0, self.n_epochs, time.time() - start_time, train_rmse, test_rmse, train_error_percent, test_error_percent)

            min_rmse = test_rmse
            min_error_percent = test_error_percent

            start_time = time.time()
            for epoch in xrange(self.n_epochs):
                count = 0
                for u, i in self.user_item_pairs:
                    count += 1
                    if count % 100000 == 0:
                        print "count-time:{0}-{1}".format(count, time.time() - start_time)
                    self.SGD_step(u, i)
                train_rmse, test_rmse = self.get_train_test_rmse()
                train_error_percent, test_error_percent = self.get_train_test_percent_error()
                train_errors.append(train_rmse)
                test_errors.append(test_rmse)
                train_error_percents.append(train_error_percent)
                test_error_percents.append(test_error_percent)
                print "[Epoch %d/%d, time %f]\ntrain error: %f, test error: %f\ntrain error percent: %f, test error percent: %f" \
                %(epoch+1, self.n_epochs, time.time() - start_time, train_rmse, test_rmse, train_error_percent, test_error_percent)
                # decrease learning rate
                self.update_learning_rate()

                min_rmse = min(min_rmse, test_rmse)
                min_error_percent = min(min_error_percent, test_error_percent)
        except KeyboardInterrupt:
            print "forced termination"

        # R = pd.DataFrame(train_X)
        # R_hat=pd.DataFrame(sgd_wr_predict(P,Q))
        # ratings = pd.DataFrame(data=R.loc[0,R.loc[0,:] > 0]).head(n=10)
        # ratings['Prediction'] = R_hat.loc[0,R.loc[0,:] > 0]
        # ratings.columns = ['Actual Rating', 'Predicted Rating']
        # print ratings

        print "recommended anime:"
        pred = self.predict_all()
        count = 0
        for i in np.argsort(pred[0])[::-1][:len(pred[0])]:
            count += 1
            print id2anime.items()[i], pred[0][i]
            if count == 100:
                break

        # error_graphs = plt.figure(figsize=(20,10))
        error_graphs = plt.figure()
        plt.suptitle(title)

        rmse_error = error_graphs.add_subplot(121)
        rmse_error.plot(range(len(train_errors)), train_errors, marker='o', label='Training Data');
        rmse_error.plot(range(len(test_errors)), test_errors, marker='v', label='Test Data');
        rmse_error.set_title('RMSE Error (min_rmse={0})'.format("%.4f" % min_rmse))
        rmse_error.set_xlabel('Number of Epochs');
        rmse_error.set_ylabel('RMSE');
        rmse_error.legend()
        rmse_error.grid()

        percent_error = error_graphs.add_subplot(122)
        percent_error.plot(range(len(train_error_percents)), train_error_percents, marker='o', label='Training Data');
        percent_error.plot(range(len(test_error_percents)), test_error_percents, marker='v', label='Test Data');
        percent_error.set_title('Percentage Error (min_error_percent={0}, correctness threshold={1})'.format("%.3f" % min_error_percent, self.threshold))
        percent_error.set_xlabel('Number of Epochs');
        percent_error.set_ylabel('Percentage Error');
        percent_error.legend()
        percent_error.grid()

        plt.show()
        print '--------------------------------------\n'

    


