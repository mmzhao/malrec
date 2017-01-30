import matplotlib.pyplot as plt
import numpy as np
import time

class Model(object):
    # Base class for all svd with sgd based models
    name = 'Generic'
    path = 'generic'

    def __init__(self, train_X, test_X, threshold, in_folder=None, out_folder=None):
        self.train_X = train_X
        self.test_X = test_X
        self.threshold = threshold
        self.in_folder = in_folder
        self.out_folder = out_folder

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

    def save_weights(self):
        pass

    def load_weights(self):
        pass

    def saved_weight_error(self):
        self.initialize_weights()
        train_rmse, test_rmse, train_error_percent, test_error_percent = self.get_train_test_both_error()
        
        print "[{0} Saved Weights]\ntrain error: {1}, test error: {2}\ntrain error percent: {3}, test error percent: {4}" \
        .format(self.name, train_rmse, test_rmse, train_error_percent, test_error_percent)

    def saved_weight_prediction(self, id2anime):
        self.initialize_weights()
        self.recommend_anime(id2anime)

    def recommend_anime(self, id2anime, num_anime=100, pred=None):
        print "recommended anime:"
        if pred is None:
            pred = self.predict_all()
        count = 0
        for i in np.argsort(pred[-1])[::-1][:len(pred[-1])]:
            count += 1
            print id2anime.items()[i], pred[-1][i]
            if count >= num_anime:
                break

    def error_hist(self, pred=None):
        # consider just doing this without the hist instead of using threshold since error seems to be gaussian
        if pred == None:
            pred = self.predict_all()
        train_errors = pred[self.train_X > 0] - self.train_X[self.train_X > 0]
        train_mean = np.mean(train_errors)
        train_std = np.std(train_errors)
        test_errors = pred[self.test_X > 0] - self.test_X[self.test_X > 0]
        test_mean = np.mean(test_errors)
        test_std = np.std(test_errors)

        hists = plt.figure(figsize=(20,10))

        train_hist = hists.add_subplot(211)
        train_hist.hist(train_errors, bins='auto')
        train_hist.set_title('Train Pred Error Histogram (mean={0}, std={1})'.format(train_mean, train_std))
        train_hist.set_xlabel('Number of Scores');
        train_hist.set_ylabel('Difference from True Score');

        test_hist = hists.add_subplot(212)
        # user_count_hist.hist(user_counts, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        test_hist.hist(test_errors, bins='auto')
        test_hist.set_title('Test Pred Error Histogram (mean={0}, std={1})'.format(test_mean, test_std))
        test_hist.set_xlabel('Number of Scores');
        test_hist.set_ylabel('Difference from True Score');

        plt.show()

    def train(self, id2anime, load=False):
        print "Training {0} model".format(self.name)

        self.initialize_weights()

        train_errors = []
        test_errors = []
        train_error_percents = []
        test_error_percents = []

        title = self.get_title()
        print title

        # TIMING
        self.time0 = 0
        self.time1 = 0
        self.time2 = 0
        self.time3 = 0
        self.time4 = 0
        # TIMING

        try:
            start_time = time.time()

            train_rmse, test_rmse, train_error_percent, test_error_percent = self.get_train_test_both_error()
            
            print "[Epoch {0}/{1}, time {2}]\ntrain error: {3}, test error: {4}\ntrain error percent: {5}, test error percent: {6}" \
            .format(0, self.n_epochs, time.time() - start_time, train_rmse, test_rmse, train_error_percent, test_error_percent)

            min_rmse = test_rmse
            min_error_percent = test_error_percent

            start_time = time.time()
            for epoch in xrange(self.n_epochs):
                count = 0
                for u, i in self.user_item_pairs:
                    count += 1
                    if count % 100000 == 0:
                        print "count-time:{0}-{1}".format(count, time.time() - start_time)
                        # TIMING
                        print "    time0:{}".format(self.time0)
                        print "    time1:{}".format(self.time1)
                        print "    time2:{}".format(self.time2)
                        print "    time3:{}".format(self.time3)
                        print "    time4:{}".format(self.time4)
                        # TIMING
                    self.SGD_step(u, i)
                print "calculating error-{0}".format(time.time() - start_time)
                train_rmse, test_rmse, train_error_percent, test_error_percent = self.get_train_test_both_error()
                train_errors.append(train_rmse)
                test_errors.append(test_rmse)
                train_error_percents.append(train_error_percent)
                test_error_percents.append(test_error_percent)
                print "[Epoch {0}/{1}, time {2}]\ntrain error: {3}, test error: {4}\ntrain error percent: {5}, test error percent: {6}" \
                .format(epoch+1, self.n_epochs, time.time() - start_time, train_rmse, test_rmse, train_error_percent, test_error_percent)
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

        if self.out_folder != None:
            self.save_weights()

        pred = self.predict_all()
        self.recommend_anime(id2anime, pred=pred)

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

    


