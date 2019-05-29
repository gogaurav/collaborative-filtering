import numpy as np
import pandas as pd
from sklearn.metrics import pairwise as pw
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.stats.stats import kendalltau

def item_item_recommend(user_shows):
    """
    Item-Item Similarity through cosine similarity(Si) and User-Item rating matrix(R)
    using the formula R * Si
    :param user_shows: User-Item rating 2D array
    :return: recommender matrix - numpy 2D array
    """
    s_i = pw.cosine_similarity(np.transpose(user_shows.values))
    return np.matmul(user_shows, s_i)


def user_user_recommend(user_shows):
    """
    User-User Similarity through cosine similarity(Su) and User-Item rating matrix(R)
    using the formula Su * R
    :param user_shows: User-Item rating 2D array
    :return: recommender matrix - numpy 2D array
    """
    s_u = pw.cosine_similarity(user_shows.values)
    return np.matmul(s_u, user_shows)


def find_top_k_shows(user, k, recommend_matrix, first_n_shows_consider=None):
    """
    Using the recommender matrix, return the top k shows
    :param user: Index of the user in User-Item rating matrix
    :param k: value of k in top k shows
    :param recommend_matrix: the matrix found out using item-item/user-user collaborative filtering method
    :param first_n_shows_consider: recommend the shows among $first_n_shows_consider shows in the dataset
    :return: list of tuples having the top k shows in descending order along with similarity scores
    """
    if not first_n_shows_consider:
        top_k_shows_idx = np.argpartition(-recommend_matrix[user], k)
    else:
        top_k_shows_idx = np.argpartition(-recommend_matrix[user][:first_n_shows_consider], k)

    top_k_shows_idx = top_k_shows_idx[:k]
    top_k_sim_scores = recommend_matrix[user][top_k_shows_idx]

    shows_sim_scores = list(zip(top_k_shows_idx, top_k_sim_scores))
    shows_sim_scores = sorted(shows_sim_scores, key=itemgetter(1), reverse=True)
    return shows_sim_scores


def find_true_positive_rate_vs_k(recommend_matrix, user_shows, user_idx, k):
    """
    Find top k shows for each k value and calculate (no of recommended shows watched)/(no of first
    100 shows watched) for each k and return it as a list
    :param recommend_matrix: matrix found out either by user-user/item-item Collaborative filtering
    :param user_shows: User-Item rating 2D array
    :param user_idx: User for which true positive rate vs k has to be found
    :param k: k is list of size 2 with range of k values to be used (passed for the range function)
    :return: list having true_positive_rate and corresponding k
    """
    true_positive_rate_vs_k = []
    no_first_100_shows_watched = np.count_nonzero(user_shows.loc[user_idx][:100].values)
    for i in range(*k):
        shows_sim_scores = find_top_k_shows(user_idx, i, recommend_matrix, 100)
        recommended_shows = [s[0] for s in shows_sim_scores]
        recommended_shows = np.array(recommended_shows)
        no_rec_shows_watched = np.count_nonzero(user_shows.loc[19][recommended_shows].values)
        true_positive_rate_vs_k.append([i, (no_rec_shows_watched / no_first_100_shows_watched)])

    return np.array(true_positive_rate_vs_k)


def plot_true_positive_rate_vs_k(usr_true_positive_rate_vs_k, itm_true_positive_rate_vs_k):
    """
    Plot line graphs for true positive rate vs k for both methods:item-item&user-user in same plot
    and save the graph in the same folder of the program files
    :param usr_true_positive_rate_vs_k:true_positive_rates and corresponding k for user-user method
    :param itm_true_positive_rate_vs_k:true_positive_rates and corresponding k for item-item method
    :return: nothing
    """
    plt.plot(usr_true_positive_rate_vs_k[:, 0], usr_true_positive_rate_vs_k[:, 1],
             label='User-User')
    plt.xticks(usr_true_positive_rate_vs_k[:, 0])
    plt.legend()
    plt.plot(itm_true_positive_rate_vs_k[:, 0], itm_true_positive_rate_vs_k[:, 1],
             label='Item-Item')
    plt.xticks(itm_true_positive_rate_vs_k[:, 0])
    plt.legend()
    plt.title("True_positive_rate VS K")
    plt.xlabel('k')
    plt.ylabel('true_positive_rate')
    plt.savefig("True_positive_rate VS K" + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def kendall_rank_correlation(item_item, user_user, itemknn, wrmf):
    """
    Find/display kendall_rank_correlation between each of the recommendation methods in the
    input parameters
    :param item_item: list having top k shows through item-item
    :param user_user: list having top k shows through user-user
    :param itemknn: list having top k shows through MyMediaLite standard library's itemknn method
    :param wrmf: list having top k shows through MyMediaLite standard library's itemknn method
    :return: nothing
    """
    recommend_types = [item_item, user_user, itemknn, wrmf]
    k_r_correlation = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if j >= i:
                k_r_correlation[i][j] = kendalltau(recommend_types[i], recommend_types[j])[0]
            else:
                k_r_correlation[i][j]= k_r_correlation[j][i]

    print("\n*** Kendall Rank correlation coefficient ***")
    table_labels = ["Item_Item", "User_User", "ItemKNN", "WRMF"]
    print("{0:^11s}{1:^11s}{2:^11s}{3:^11s}{4:^11s}".format("", *table_labels))
    for i in range(4):
        print("{0:11s}".format(table_labels[i]), end='')
        for j in range(4):
            print("{0:^11.5f}".format(k_r_correlation[i][j]), end='')
        print()






