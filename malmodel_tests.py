import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def svd_plus2_test(train_X, test_X, id2anime):

    def rmse(Im,R,mu,Bu,Bi,P,Q,Y,I,N):
        # inefficient
        return np.sqrt(np.sum((Im * (R - predict_all(mu,Bu,Bi,P,Q,Y,I,N)))**2)/len(R[R > 0]))

    def rmse_train(R,mu,Bu,Bi,P,Q,Y,I,N):
        return np.sqrt(sq_error_train(R,mu,Bu,Bi,P,Q,Y,I,N)/len(R[R > 0]))
        # b = np.sqrt(np.sum((Im * (R - predict_train(P,Q,Y,I,N)))**2)/len(R[R > 0]))

    def rmse_test(R,mu,Bu,Bi,P,Q,Y,I,N):
        return np.sqrt(sq_error_test(R,mu,Bu,Bi,P,Q,Y,I,N)/len(R[R > 0]))
        # return np.sqrt(np.sum((Im * (R - predict_test(P,Q,Y,I,N)))**2)/len(R[R > 0]))

    def predict_single(mu,Bu,Bi,P,Q,Y,Iu,N):
        # if N:
        return mu + Bu + Bi + np.dot(Q, P + N**-.5 * np.dot(Y.T, Iu))
        # return np.dot(Q, P)

    def predict_all(mu,Bu,Bi,P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u in range(m):
            for i in range(n):
                pred[u, i] = predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u])
        return pred

    def predict_train(mu,Bu,Bi,P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u, i in zip(users,items):
            pred[u, i] = predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u])
        return pred

    def predict_test(mu,Bu,Bi,P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u, i in zip(users_test,items_test):
            pred[u, i] = predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u])
        return pred

    def sq_error_train(R,mu,Bu,Bi,P,Q,Y,I,N):
        # cur = time.time()
        sq_err = 0
        for u, i in zip(users,items):
            sq_err += (R[u, i] - predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u]))**2
        # print "inb4", time.time() - cur
        return sq_err

    def sq_error_test(R,mu,Bu,Bi,P,Q,Y,I,N):
        sq_err = 0
        for u, i in zip(users_test,items_test):
            sq_err += (R[u, i] - predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u]))**2
        return sq_err

    def percent_error_train(R,mu,Bu,Bi,P,Q,Y,I,N,threshold):
        wrong = 0
        for u, i in zip(users,items):
            if abs(R[u, i] - predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u])) > threshold:
                wrong += 1
        return float(wrong)/len(users)

    def percent_error_test(R,mu,Bu,Bi,P,Q,Y,I,N,threshold):
        wrong = 0
        for u, i in zip(users_test,items_test):
            if abs(R[u, i] - predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u])) > threshold:
                wrong += 1
        return float(wrong)/len(users_test)

    print "starting svd++ test"

    m, n = train_X.shape  # Number of users and items

    #Only consider non-zero matrix 
    users,items = train_X.nonzero()
    users_test,items_test = test_X.nonzero()
    user_item_pairs = zip(users,items)

    mu = np.sum(train_X) / len(user_item_pairs) # total mean of all scores
    # try initializing with dif between user/anime mean and total mean
    Bu = 1 * np.random.rand(m) # user biases
    Bi = 1 * np.random.rand(n) # anime biases

    I = train_X.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    N = np.sum(train_X, axis=1)
    N[N == 0] = 1 #debatable, either need this or put ifs back

    I2 = test_X.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0

    # closest test/train error is (k=20,gamma1=.007,gamma2=.005,gammaX=.97/.95,lmbda1=.050,lmbda2=.050)
    lmbda1 = 0.050 # Regularization weight for Bu,Bi
    lmbda2 = 0.050 # Regularization weight for P/Q/Y
    gamma1=0.007  # Learning rate for Bu/Bi 
    gamma2=0.005  # Learning rate for P/Q/Y
    gammaX = .95 # Try other learning rate adaptation methods (bold driver, annealing, search-then-converge)
    g1 = gamma1 # Later for the graph
    g2 = gamma2 # Later for the graph
    k = 20  # Dimensionality of the latent feature space
    threshold = .5
    n_epochs = 100  # Number of epochs

    P = .1 * np.random.rand(m,k) # Latent user feature matrix
    Q = .1 * np.random.rand(n,k) # Latent movie feature matrix
    Y = .1 * np.random.rand(n,k) # Implicit feedback feature matrix

    train_errors = []
    test_errors = []
    train_error_percents = []
    test_error_percents = []

    # cur = time.time()
    # a = rmse_train(I,train_X,P,Q,Y,I,N) # Calculate root mean squared error from train dataset
    # print "train time:", time.time() - cur, a
    # b = rmse_test(I2,test_X,P,Q,Y,I,N) # Calculate root mean squared error from test dataset
    # print "test time:", time.time() - cur, b
    # exit(1)

    time0 = 0
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0

    print 'SVD++ Learning Curve (k={0},gmma={1},lmbda={2}), correctness threshold={3}'.format(k,(g1,g2,gammaX),(lmbda1,lmbda2), threshold)

    try:
        start_time = time.time()

        train_rmse = rmse_train(train_X,mu,Bu,Bi,P,Q,Y,I,N) # Calculate root mean squared error from train dataset
        # print time.time() - cur
        test_rmse = rmse_test(test_X,mu,Bu,Bi,P,Q,Y,I,N) # Calculate root mean squared error from test dataset
        # print time.time() - cur
        # train_errors.append(train_rmse)
        # test_errors.append(test_rmse)
        train_error_percent = percent_error_train(train_X,mu,Bu,Bi,P,Q,Y,I,N,threshold)
        test_error_percent = percent_error_test(test_X,mu,Bu,Bi,P,Q,Y,I,N,threshold)

        print "[Epoch %d/%d, time %f]\ntrain error: %f, test error: %f\ntrain error percent: %f, test error percent: %f" \
        %(0, n_epochs, time.time() - start_time, train_rmse, test_rmse, train_error_percent, test_error_percent)

        min_rmse = test_rmse
        min_error_percent = test_error_percent
            
        for epoch in xrange(n_epochs):
            count = 0
            for u, i in user_item_pairs:

                # cur = time.time()

                # pred_train = predict_train(P,Q,Y,I,N)

                # time0 += time.time() - cur

                count += 1
                if count % 100000 == 0:
                    print "count-time:{0}-{1}".format(count, time.time() - start_time)
                    # train_rmse = rmse_train(I,train_X,mu,Bu,Bi,P,Q,Y,I,N) # Calculate root mean squared error from train dataset
                    # test_rmse = rmse_test(I2,test_X,mu,Bu,Bi,P,Q,Y,I,N) # Calculate root mean squared error from test dataset
                    # print "train rmse:", train_rmse
                    # print "test rmse:", test_rmse
                    # print "    time0:{}".format(time0)
                    # print "    time1:{}".format(time1)
                    # print "    time2:{}".format(time2)
                    # print "    time3:{}".format(time3)
                    # print "    time4:{}".format(time4)
                # cur = time.time()

                # e = train_X[u, i] - pred_train[u, i]  # Calculate error for gradient
                e = train_X[u, i] - predict_single(mu,Bu[u],Bi[i],P[u],Q[i],Y,I[u],N[u])  # Calculate error for gradient
                
                # time0 += time.time() - cur
                # cur = time.time()

                Bu[u] += gamma1 * (e - lmbda1 * Bu[u])
                Bu[i] += gamma1 * (e - lmbda1 * Bu[i])

                # time1 += time.time() - cur
                # cur = time.time()
                
                # if N[u]:
                Q[i] += gamma2 * (e * (P[u] + N[u]**-.5 * np.dot(Y.T, I[u])) - lmbda2 * Q[i])  # Update latent movie feature matrix
                # else:
                    # Q[i] += gamma * (e * (P[u] + np.dot(Y.T, I[u])) - lmbda * Q[i])  # Update latent movie feature matrix
                
                # time2 += time.time() - cur
                # cur = time.time()

                P[u] += gamma2 * (e * Q[i] - lmbda2 * P[u]) # Update latent user feature matrix
                
                # time3 += time.time() - cur
                # cur = time.time()

                # this literally optimized runtime by 1000x
                Nu = train_X[u].nonzero()[0]
                Y[Nu] *= (1 - gamma2 * lmbda2)
                # if N[u]:
                Yd = gamma2 * (e * N[u]**-.5 * Q[i])
                # else:
                    # Yd = gamma * (e * Q[i])
                np.add(Y[Nu], Yd)
                # for j in train_X[u].nonzero()[0]:
                    # Y[j] += gamma * (e * N[u]**-.5 * Q[i] - lmbda * Y[j])

                # time4 += time.time() - cur

                # print sum(sum(P))
                # print sum(sum(Q))
            # cur = time.time()
            train_rmse = rmse_train(train_X,mu,Bu,Bi,P,Q,Y,I,N) # Calculate root mean squared error from train dataset
            # print time.time() - cur
            test_rmse = rmse_test(test_X,mu,Bu,Bi,P,Q,Y,I,N) # Calculate root mean squared error from test dataset
            # print time.time() - cur
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)
            train_error_percent = percent_error_train(train_X,mu,Bu,Bi,P,Q,Y,I,N,threshold)
            test_error_percent = percent_error_test(test_X,mu,Bu,Bi,P,Q,Y,I,N,threshold)
            train_error_percents.append(train_error_percent)
            test_error_percents.append(test_error_percent)
            print "[Epoch %d/%d, time %f]\ntrain error: %f, test error: %f\ntrain error percent: %f, test error percent: %f" \
            %(epoch+1, n_epochs, time.time() - start_time, train_rmse, test_rmse, train_error_percent, test_error_percent)
            # decrease learning rate
            gamma1 *= gammaX
            gamma2 *= gammaX

            min_rmse = min(min_rmse, test_rmse)
            min_error_percent = min(min_error_percent, test_error_percent)
    # R = pd.DataFrame(train_X)
    # R_hat=pd.DataFrame(sgd_wr_predict(P,Q))
    # ratings = pd.DataFrame(data=R.loc[0,R.loc[0,:] > 0]).head(n=10)
    # ratings['Prediction'] = R_hat.loc[0,R.loc[0,:] > 0]
    # ratings.columns = ['Actual Rating', 'Predicted Rating']
    # print ratings
    except KeyboardInterrupt:
        print "forced termination"

    print "recommended anime:"
    pred = predict_all(mu,Bu,Bi,P,Q,Y,I,N)
    count = 0
    for i in np.argsort(pred[0])[::-1][:len(pred[0])]:
        count += 1
        print id2anime.items()[i], pred[0][i]
        if count == 100:
            break

    # error_graphs = plt.figure(figsize=(20,10))
    error_graphs = plt.figure()
    plt.suptitle('SVD++ Learning Curves (k={0},gmma={1},lmbda={2})'.format(k,(g1,g2,gammaX),(lmbda1,lmbda2)))

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
    percent_error.set_title('Percentage Error (min_error_percent={0}, correctness threshold={1})'.format("%.3f" % min_error_percent, threshold))
    percent_error.set_xlabel('Number of Epochs');
    percent_error.set_ylabel('Percentage Error');
    percent_error.legend()
    percent_error.grid()

    plt.show()
    print '--------------------------------------\n'

def svd_plus2_simplified_test(train_X, test_X, id2anime):
    # https://mahout.apache.org/users/recommender/matrix-factorization.html

    def rmse(Im,R,P,Q,Y,I,N):
        # inefficient
        return np.sqrt(np.sum((Im * (R - predict_all(P,Q,Y,I,N)))**2)/len(R[R > 0]))

    def rmse_train(Im,R,P,Q,Y,I,N):
        return np.sqrt(sq_error_train(R,P,Q,Y,I,N)/len(R[R > 0]))
        # b = np.sqrt(np.sum((Im * (R - predict_train(P,Q,Y,I,N)))**2)/len(R[R > 0]))

    def rmse_test(Im,R,P,Q,Y,I,N):
        return np.sqrt(sq_error_test(R,P,Q,Y,I,N)/len(R[R > 0]))
        # return np.sqrt(np.sum((Im * (R - predict_test(P,Q,Y,I,N)))**2)/len(R[R > 0]))

    def predict_single(P,Q,Y,Iu,N):
        # if N:
        return np.dot(Q, P + N**-.5 * np.dot(Y.T, Iu))
        # return np.dot(Q, P)

    def predict_all(P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u in range(m):
            for i in range(n):
                pred[u, i] = predict_single(P[u],Q[i],Y,I[u],N[u])
        return pred

    def predict_train(P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u, i in zip(users,items):
            pred[u, i] = predict_single(P[u],Q[i],Y,I[u],N[u])
        return pred

    def predict_test(P,Q,Y,I,N):
        pred = np.zeros(train_X.shape)
        for u, i in zip(users_test,items_test):
            pred[u, i] = predict_single(P[u],Q[i],Y,I[u],N[u])
        return pred

    def sq_error_train(R,P,Q,Y,I,N):
        # cur = time.time()
        sq_err = 0
        for u, i in zip(users,items):
            sq_err += (R[u, i] - predict_single(P[u],Q[i],Y,I[u],N[u]))**2
        # print "inb4", time.time() - cur
        return sq_err

    def sq_error_test(R,P,Q,Y,I,N):
        sq_err = 0
        for u, i in zip(users_test,items_test):
            sq_err += (R[u, i] - predict_single(P[u],Q[i],Y,I[u],N[u]))**2
        return sq_err

    print "starting svd++ simplified test"

    #Only consider non-zero matrix 
    users,items = train_X.nonzero()
    users_test,items_test = test_X.nonzero()
    user_item_pairs = zip(users,items)

    I = train_X.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    N = np.sum(train_X, axis=1)
    N[N == 0] = 1 #debatable, either need this or put ifs back

    I2 = test_X.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0

    lmbda = 0.01 # Regularization weight for P/Q/Y
    k = 20  # Dimensionality of the latent feature space
    m, n = train_X.shape  # Number of users and items
    n_epochs = 200  # Number of epochs
    gamma=0.005  # Learning rate

    P = 1 * np.random.rand(m,k) # Latent user feature matrix
    Q = 1 * np.random.rand(n,k) # Latent movie feature matrix
    Y = .1 * np.random.rand(n,k) # Implicit feedback feature matrix

    train_errors = []
    test_errors = []

    # cur = time.time()
    # a = rmse_train(I,train_X,P,Q,Y,I,N) # Calculate root mean squared error from train dataset
    # print "train time:", time.time() - cur, a
    # b = rmse_test(I2,test_X,P,Q,Y,I,N) # Calculate root mean squared error from test dataset
    # print "test time:", time.time() - cur, b
    # exit(1)

    time0 = 0
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0
    # print len(zip(users,items))
    start_time = time.time()
    for epoch in xrange(n_epochs):
        count = 0
        for u, i in user_item_pairs:

            # cur = time.time()

            # pred_train = predict_train(P,Q,Y,I,N)

            # time0 += time.time() - cur

            # count += 1
            # if count % 100000 == 0:
                # print "count-time:{0}-{1}".format(count, time.time() - start_time)
                # train_rmse = rmse_train(I,train_X,P,Q,Y,I,N) # Calculate root mean squared error from train dataset
                # test_rmse = rmse_test(I2,test_X,P,Q,Y,I,N) # Calculate root mean squared error from test dataset
                # print "train rmse:", train_rmse
                # print "test rmse:", test_rmse
                # print "    time1:{}".format(time1)
                # print "    time2:{}".format(time2)
                # print "    time3:{}".format(time3)
                # print "    time4:{}".format(time4)
            # cur = time.time()

            # e = train_X[u, i] - pred_train[u, i]  # Calculate error for gradient
            e = train_X[u, i] - predict_single(P[u],Q[i],Y,I[u],N[u])  # Calculate error for gradient
            
            # time1 += time.time() - cur
            # cur = time.time()
            
            # if N[u]:
            Q[i] += gamma * (e * (P[u] + N[u]**-.5 * np.dot(Y.T, I[u])) - lmbda * Q[i])  # Update latent movie feature matrix
            # else:
                # Q[i] += gamma * (e * (P[u] + np.dot(Y.T, I[u])) - lmbda * Q[i])  # Update latent movie feature matrix
            
            # time2 += time.time() - cur
            # cur = time.time()

            P[u] += gamma * (e * Q[i] - lmbda * P[u]) # Update latent user feature matrix
            
            # time3 += time.time() - cur
            # cur = time.time()

            # this literally optimized runtime by 1000x
            Nu = train_X[u].nonzero()[0]
            Y[Nu] *= (1 - gamma * lmbda)
            # if N[u]:
            Yd = gamma * (e * N[u]**-.5 * Q[i])
            # else:
                # Yd = gamma * (e * Q[i])
            Y[nu] = np.add(Y[Nu], Yd)
            # for j in train_X[u].nonzero()[0]:
                # Y[j] += gamma * (e * N[u]**-.5 * Q[i] - lmbda * Y[j])

            # time4 += time.time() - cur

            # print sum(sum(P))
            # print sum(sum(Q))
        # cur = time.time()
        train_rmse = rmse_train(I,train_X,P,Q,Y,I,N) # Calculate root mean squared error from train dataset
        # print time.time() - cur
        test_rmse = rmse_test(I2,test_X,P,Q,Y,I,N) # Calculate root mean squared error from test dataset
        # print time.time() - cur
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        print "[Epoch %d/%d, time %f] train error: %f, test error: %f" \
        %(epoch+1, n_epochs, time.time() - start_time, train_rmse, test_rmse)
        # decrease learning rate
        gamma *= .95
    # R = pd.DataFrame(train_X)
    # R_hat=pd.DataFrame(sgd_wr_predict(P,Q))
    # ratings = pd.DataFrame(data=R.loc[0,R.loc[0,:] > 0]).head(n=10)
    # ratings['Prediction'] = R_hat.loc[0,R.loc[0,:] > 0]
    # ratings.columns = ['Actual Rating', 'Predicted Rating']
    # print ratings

    print "recommended anime:"
    pred = predict_all(P,Q,Y,I,N)
    count = 0
    for i in np.argsort(pred[0])[::-1][:len(pred[0])]:
        count += 1
        print id2anime.items()[i], pred[0][i]
        if count == 100:
            break

    plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
    plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
    plt.title('SVD++ Simplified Learning Curve')
    plt.xlabel('Number of Epochs');
    plt.ylabel('RMSE');
    plt.legend()
    plt.grid()
    plt.show()
    print '--------------------------------------\n'

def als_test(train_X, test_X, id2anime):
    # alternating least squares
    def als_rmse(I,R,P,Q):
        return np.sqrt(np.sum((I * (R - als_predict(P,Q)))**2)/len(R[R > 0]))

    def als_predict(P,Q):
        return np.dot(P.T,Q)

    print "starting als test"
    # Index matrix for training data
    I = train_X.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    # Index matrix for test data
    I2 = test_X.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0

    lmbda = 0.1 # Regularisation weight
    k = 20 # Dimensionality of latent feature space
    m, n = train_X.shape # Number of users and items
    n_epochs = 5 # Number of epochs

    P = 3 * np.random.rand(k,m) # Latent user feature matrix
    Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
    Q[0,:] = train_X[train_X != 0].mean(axis=0) # Avg. rating for each movie
    E = np.eye(k) # (k x k)-dimensional identity matrix

    train_errors = []
    test_errors = []

    start_time = time.time()

    # Repeat until convergence
    for epoch in range(n_epochs):
        # Fix Q and estimate P
        for i, Ii in enumerate(I):
            nui = np.count_nonzero(Ii) # Number of items user i has rated
            if (nui == 0): nui = 1 # Be aware of zero counts!
        
            # Least squares solution
            Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E
            Vi = np.dot(Q, np.dot(np.diag(Ii), train_X[i].T))
            P[:,i] = np.linalg.solve(Ai,Vi)
            
        # Fix P and estimate Q
        for j, Ij in enumerate(I.T):
            nmj = np.count_nonzero(Ij) # Number of users that rated item j
            if (nmj == 0): nmj = 1 # Be aware of zero counts!
            
            # Least squares solution
            Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
            Vj = np.dot(P, np.dot(np.diag(Ij), train_X[:,j]))
            Q[:,j] = np.linalg.solve(Aj,Vj)
        
        train_rmse = als_rmse(I,train_X,P,Q)
        test_rmse = als_rmse(I2,test_X,P,Q)
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        
        print "time spent:", time.time() - start_time
        print "[Epoch %d/%d] train error: %f, test error: %f" \
        %(epoch+1, n_epochs, train_rmse, test_rmse)

    R_hat = pd.DataFrame(als_predict(P,Q))
    R = pd.DataFrame(train_X)
    ratings = pd.DataFrame(data=R.loc[0,R.loc[0,:] > 0]).head(n=10)
    ratings['Prediction'] = R_hat.loc[0,R.loc[0,:] > 0]
    ratings.columns = ['Actual Rating', 'Predicted Rating']
    print ratings

    # print "recommended anime:"
    # pred = als_predict(P,Q)
    # count = 0
    # for i in np.argsort(pred[0])[::-1][:len(pred[0])]:
    #     count += 1
    #     print id2anime.items()[i], pred[0][i]
    #     if count == 100:
    #         break

    plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
    plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
    plt.title('ALS-WR Learning Curve')
    plt.xlabel('Number of Epochs');
    plt.ylabel('RMSE');
    plt.legend()
    plt.grid()
    plt.show()
    print '--------------------------------------\n'


def sgd_wr_test(train_X, test_X, id2anime):
    # stochastic gradient descent with weighted lambda regularisation
    def sgd_rmse(I,R,P,Q):
        # print P
        # print Q
        # print sgd_wr_predict(P,Q)
        # print (R - sgd_wr_predict(P,Q))
        # print (I * (R - sgd_wr_predict(P,Q)))
        # print len(R[R > 0])
        # print (I * (R - sgd_wr_predict(P,Q)))**2
        # print ((I * (R - sgd_wr_predict(P,Q)))**2)/len(R[R > 0])
        # print np.sum((I * (R - sgd_wr_predict(P,Q)))**2)/len(R[R > 0])
        return np.sqrt(np.sum((I * (R - sgd_wr_predict(P,Q)))**2)/len(R[R > 0]))

    def sgd_wr_predict(P,Q):
        return np.dot(P.T,Q)

    print "starting sgd-wr test"
    I = train_X.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    I2 = test_X.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0

    lmbda = 0.1 # Regularisation weight
    k = 20  # Dimensionality of the latent feature space
    m, n = train_X.shape  # Number of users and items
    n_epochs = 50  # Number of epochs
    gamma=0.005  # Learning rate

    P = 3 * np.random.rand(k,m) # Latent user feature matrix
    Q = 3 * np.random.rand(k,n) # Latent movie feature matrix

    train_errors = []
    test_errors = []

    #Only consider non-zero matrix 
    users,items = train_X.nonzero()
    user_item_pairs = zip(users,items)
    start_time = time.time()
    for epoch in xrange(n_epochs):
        for u, i in user_item_pairs:
            e = train_X[u, i] - sgd_wr_predict(P[:,u],Q[:,i])  # Calculate error for gradient
            P[:,u] += gamma * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
            Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix
            # print sum(sum(P))
            # print sum(sum(Q))
        train_rmse = sgd_rmse(I,train_X,P,Q) # Calculate root mean squared error from train dataset
        test_rmse = sgd_rmse(I2,test_X,P,Q) # Calculate root mean squared error from test dataset
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        print "[Epoch %d/%d, time %f] train error: %f, test error: %f" \
        %(epoch+1, n_epochs, time.time() - start_time, train_rmse, test_rmse)

    # R = pd.DataFrame(train_X)
    # R_hat=pd.DataFrame(sgd_wr_predict(P,Q))
    # ratings = pd.DataFrame(data=R.loc[0,R.loc[0,:] > 0]).head(n=10)
    # ratings['Prediction'] = R_hat.loc[0,R.loc[0,:] > 0]
    # ratings.columns = ['Actual Rating', 'Predicted Rating']
    # print ratings

    print "recommended anime:"
    pred = sgd_wr_predict(P,Q)
    count = 0
    for i in np.argsort(pred[0])[::-1][:len(pred[0])]:
        count += 1
        print id2anime.items()[i], pred[0][i]
        if count == 100:
            break

    plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
    plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
    plt.title('SGD-WR Learning Curve')
    plt.xlabel('Number of Epochs');
    plt.ylabel('RMSE');
    plt.legend()
    plt.grid()
    plt.show()
    print '--------------------------------------\n'


def svd_test(train_X, test_X, id2anime):
    # svd matrix factorization is model based collaborative filtering

    def rmse(pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return math.sqrt(mean_squared_error(pred, actual))

    print "starting svd test"
    k = 100
    u, s, vt = sparse.linalg.svds(train_X, k=k)
    S_diag=np.diag(s)
    pred = np.dot(np.dot(u, S_diag), vt)
    print 'SVD RMSE: ' + str(rmse(pred, test_X))
    # print "recommended anime:"
    # count = 0
    # for i in np.argsort(pred[0])[::-1][:len(pred[0])]:
    #     count += 1
    #     print id2anime.items()[i], pred[0][i]
    #     if count == 100:
    #         break
    print '--------------------------------------\n'

def cos_sim_test(train_X, test_X, id2anime):
    # cosine similarity is memory based collaborative filtering
    def cos_sim_predict(ratings, similarity, type='user'):
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
        return pred

    def rmse(pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return math.sqrt(mean_squared_error(pred, actual))

    print "starting cosine similarity test"
    user_similarity = pairwise_distances(train_X, metric='cosine')
    item_similarity = pairwise_distances(train_X.T, metric='cosine')
    item_prediction = cos_sim_predict(train_X, item_similarity, type='item')
    user_prediction = cos_sim_predict(train_X, user_similarity, type='user')
    print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_X))
    # print "recommended anime:"
    # count = 0
    # for i in np.argsort(user_prediction[0])[::-1][:len(user_prediction[0])]:
    #     count += 1
    #     print id2anime.items()[i], user_prediction[0][i]
    #     if count == 100:
    #         break
    print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_X))
    # print "recommended anime:"
    # count = 0
    # for i in np.argsort(item_prediction[0])[::-1][:len(item_prediction[0])]:
    #     count += 1
    #     print id2anime.items()[i], item_prediction[0][i]
    #     if count == 100:
    #         break
    print '--------------------------------------\n'

