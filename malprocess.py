# MODEL BASED COLLABORATIVE FILTERING ANIME RECOMMENDATION SYSTEM
import collections
import json
import malscrape
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy as sp
import scipy.sparse as sparse
import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

import malmodel_tests

from svd import SVD
from svdplusplus_simplified import SVDPlusPlusSimplified
from svdplusplus import SVDPlusPlus
from svdplusplus_cuda import SVDPlusPlusCuda
from asymmetric_svd import AsymmetricSVD

PROCESSED_DATA_FOLDER = 'processed_data'

# TODO: currently using sorted lists, time against using dicts
def makeArrays(start, end, animelists, id2anime, score_map=None, outfolder=None):
    if score_map == None:
        score_map = [i for i in range(11)]
    score_array = None
    users = animelists.keys()
    score_count = 0
    unused_lists = 0
    all_anime_sorted = id2anime.keys()
    actual_users = []
    user_to_row = {}
    start_time = time.time()
    for i in range(start, min(end, len(users))):
        if (i+1)%1000 == 0:
            print 'finished processing {} animelists, time spent {}'.format(i+1, time.time() - start_time)
        # print animelists[users[i]]
        anime2score = animelists[users[i]]
        anime2score = {int(k):v for k,v in anime2score.items()}
        user_anime_sorted = anime2score.keys()
        user_anime_sorted.sort()
        full_scores = []
        user_index = 0
        all_index = 0
        scores_1000 = 0
        while user_index < len(user_anime_sorted) and all_index < len(all_anime_sorted):
            # print user_index, all_index, user_anime_sorted[user_index], all_anime_sorted[all_index]
            if user_anime_sorted[user_index] == all_anime_sorted[all_index]:
                full_scores += [score_map[anime2score[user_anime_sorted[user_index]]]]
                user_index += 1
                all_index += 1
                # maybe deal with already watched but unscored animes differently
                if full_scores[-1]:
                    scores_1000 += 1
            elif user_anime_sorted[user_index] not in all_anime_sorted:
                # print 'anime not in top 1000 most popular:', user_anime_sorted[user_index] #, id2anime[user_anime_sorted[user_index]]
                user_index += 1
            else: # user_anime_sorted[user_index] > all_anime_sorted[all_index]:
                full_scores += [0]
                all_index += 1
        full_scores += [0 for _ in range(len(all_anime_sorted) - len(full_scores))]
        # print full_scores
        if scores_1000 == 0:
            unused_lists += 1
            continue
            # print "list unused:", users[i], i
        elif score_array is not None:
            score_array = np.append(score_array, [full_scores], axis=0)
        else:
            score_array = np.array([full_scores])
        # print 'scores in top 1000 most popular:', scores_1000
        # print score_array.shape
        user_to_row[users[i]] = len(actual_users)
        actual_users += [users[i]]
        score_count += scores_1000
    print 'total time to make array:', time.time() - start_time
    print 'lists converted to array:', min(end, len(users)) - start - unused_lists
    print 'unused lists:', unused_lists
    print 'average scores/user:', float(score_count)/(float(min(end, len(users)) - start - unused_lists))
    # print score_array.shape

    if not os.path.exists("{}/{}".format(PROCESSED_DATA_FOLDER, outfolder)):
        os.makedirs("{}/{}".format(PROCESSED_DATA_FOLDER, outfolder))
    header = ','.join([id2anime[aid] for aid in all_anime_sorted])
    header = ''.join([i if ord(i) < 128 else '' for i in header])
    # print headers
    user_info = {}
    user_info['usernames'] = actual_users
    user_info['order_map'] = user_to_row
    with open("{}/{}/user_info.json".format(PROCESSED_DATA_FOLDER, outfolder), 'w') as f:
        json.dump(user_info, f, indent=2)
    np.savetxt("{}/{}/scores.csv".format(PROCESSED_DATA_FOLDER, outfolder), score_array, fmt='%d', delimiter=',', header=header)
    # np.savetxt(outfile, score_array, fmt='%d', delimiter=',')

def single_user_pipeline(username, id2anime, score_map=None):
    malscrape.scrapeAnimelistsNew([username], 1, outfile="{}_animelist.json".format(username))
    userlist = malscrape.getAnimelists("{}_animelist.json".format(username))
    makeArrays(0, 1, userlist, id2anime, score_map=score_map, outfolder=username)
 
def recommend(score_array, user_array, id2anime):
    score_array = np.concatenate((score_array, user_array), axis=0)
    # norm_score_array = zscale(score_array)
    # norm_score_array = normalizeUsersAnime(score_array)
    norm_score_array = score_array
    # print norm_score_array[0]
    # print score_array.shape
    num_users = score_array.shape[0]
    num_anime = score_array.shape[1]
    # score_indicies = np.nonzero(score_array)
    # score_indicies = np.array([score_indicies[0], score_indicies[1]]).T
    score_indicies = np.nonzero(norm_score_array)
    score_indicies = np.array([score_indicies[0], score_indicies[1]]).T

    sparsity = round(1.0 - len(score_indicies) / float(num_users*num_anime),3)
    print 'The sparsity level of MAL data is ' +  str(sparsity*100) + '%'

    # train_data, test_data = train_test_split(score_indicies, test_size=0.20)
    train_data, test_data = train_test_split(score_indicies, test_size=0.20, random_state=0)

    train_X = np.zeros((num_users, num_anime), dtype=float)
    for line in train_data:
        train_X[line[0], line[1]] = norm_score_array[line[0], line[1]]
        # train_X[line[0], line[1]] = score_array[line[0], line[1]]

    test_X = np.zeros((num_users, num_anime), dtype=float)
    for line in test_data:
        test_X[line[0], line[1]] = norm_score_array[line[0], line[1]]
        # test_X[line[0], line[1]] = score_array[line[0], line[1]]

    # malmodel_tests.cos_sim_test(train_X, test_X, id2anime)
    # malmodel_tests.svd_test(train_X, test_X, id2anime)
    # malmodel_tests.sgd_wr_test(train_X, test_X, id2anime)
    # malmodel_tests.als_test(train_X, test_X, id2anime)
    # malmodel_tests.svd_plus2_simplified_test(train_X, test_X, id2anime)
    # malmodel_tests.svd_plus2_test(train_X, test_X, id2anime)

    # SVD(train_X, test_X).train(id2anime)
    # SVDPlusPlusSimplified(train_X, test_X).train(id2anime)
    # SVDPlusPlusCuda(train_X, test_X).train(id2anime)


    SVDPlusPlus(train_X, test_X).train(id2anime)
    # SVDPlusPlus(train_X, test_X, in_folder="bear", out_folder="bear").train(id2anime)
    # SVDPlusPlus(train_X, test_X, in_folder=None, out_folder="real75").train(id2anime)
    # SVDPlusPlus(train_X, test_X, in_folder="test1", out_folder="test1").saved_weight_error()
    # SVDPlusPlus(train_X, test_X, in_folder="test1", out_folder="test1").saved_weight_prediction(id2anime)
    
    # svdpp = SVDPlusPlus(train_X, test_X, in_folder="real", out_folder=None)
    # svdpp.initialize_weights()
    # svdpp.error_hist()

    # AsymmetricSVD(train_X, test_X).train(id2anime)


def zscale(X):
    # gaussian normalization
    means = np.average(X, axis=1, weights=X.astype(bool))
    # print means[:10]
    zX = X.astype(float)
    for i in range(X.shape[0]):
        for j in np.nonzero(zX[i])[0]:
            zX[i][j] = (zX[i][j] - means[i])
    # print np.average(zX, axis=1, weights=X.astype(bool))
    stds = np.sqrt(np.average((zX)**2, axis=1, weights=X.astype(bool)))
    # print stds
    for i in range(X.shape[0]):
        for j in np.nonzero(zX[i])[0]:
            zX[i][j] = zX[i][j] / stds[i]
    # print np.average(zX, axis=1, weights=X.astype(bool))
    # print np.sqrt(np.average((zX)**2, axis=1, weights=X.astype(bool)))
    
    # TODO: impute values based on mode or something else
    # print sp.stats.mode(zX).shape
    # print sp.stats.mode(zX)
    return zX

def normalizeUsers(X):
    means = np.average(X, axis=1, weights=X.astype(bool))
    # print means[:10]
    zX = X.astype(float)
    for i in range(X.shape[0]):
        for j in np.nonzero(zX[i])[0]:
            zX[i][j] = (zX[i][j] - means[i])
    # print np.average(zX, axis=1, weights=X.astype(bool))
    return zX

def normalizeAnime(X):
    means = np.average(X, axis=0, weights=X.astype(bool))
    # print means[:10]
    zX = X.astype(float)
    for i in range(X.shape[0]):
        for j in np.nonzero(zX[i])[0]:
            zX[i][j] = (zX[i][j] - means[j])
    # print np.average(zX, axis=1, weights=X.astype(bool))
    return zX

def normalizeUsersAnime(X):
    return normalizeAnime(normalizeUsers(X))

def normalizeAnimeUsers(X):
    return normalizeUsers(normalizeAnime(X))

def stats(score_array, id2anime):
    m = score_array.copy()
    m[m>0] = 1

    hists = plt.figure(figsize=(20,10))

    anime_counts = np.sum(m, axis=0)
    # for i in range(len(anime_counts)):
        # print id2anime.items()[i], anime_counts[i]
    anime_count_hist = hists.add_subplot(211)
    anime_count_hist.hist(anime_counts, bins='auto')
    anime_count_hist.set_title('Anime Score Count Histogram')
    anime_count_hist.set_xlabel('Number of Scores');
    anime_count_hist.set_ylabel('Number of Animes');

    user_counts = np.sum(m, axis=1)
    # for i in range(len(user_counts)):
        # print id2anime.items()[i], user_counts[i]
    user_count_hist = hists.add_subplot(212)
    # user_count_hist.hist(user_counts, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    user_count_hist.hist(user_counts, bins='auto')
    user_count_hist.set_title('User Score Count Histogram')
    user_count_hist.set_xlabel('Number of Scores');
    user_count_hist.set_ylabel('Number of Users');

    plt.show()

if __name__ == "__main__":
    id2anime = malscrape.getId2AnimeDictSorted()
    # single_user_pipeline("AFreakingBear", id2anime)
    # single_user_pipeline("ploebian", id2anime)
    # exit(0)

    # animelists = malscrape.getAnimelists('user_animelists_club.json')
    # animelists = malscrape.getAnimelists('user_animelists_club_10000.json')

    # makeArrays(0, 100000, animelists, id2anime, outfolder='anime_1000_club_100000')
    # makeArrays(0, 200000, animelists, id2anime, outfolder='anime_1000_club_200000')
    
    score_data = pd.DataFrame.from_csv(PROCESSED_DATA_FOLDER + "/anime_1000_club_10000/scores.csv", index_col=False)
    score_array = score_data.values
    pleb_data = pd.DataFrame.from_csv(PROCESSED_DATA_FOLDER + "/ploebian/scores.csv", index_col=False)
    pleb_array = pleb_data.values
    bear_data = pd.DataFrame.from_csv(PROCESSED_DATA_FOLDER + "/AFreakingBear/scores.csv", index_col=False)
    bear_array = bear_data.values
    # stats(score_array, id2anime)
    recommend(score_array, pleb_array, id2anime)
    # recommend(score_array, bear_array, id2anime)



