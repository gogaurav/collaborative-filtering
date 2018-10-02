import numpy as np
from sklearn.metrics import pairwise as pw
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.stats.stats import kendalltau


def item_item_recommend(user_shows):
    s_i = pw.cosine_similarity(np.transpose(user_shows))
    return np.matmul(user_shows, s_i)


def user_user_recommend(user_shows):
    s_u = pw.cosine_similarity(user_shows)
    return np.matmul(s_u, user_shows)


def find_top_k_shows(user, k, recommend_matrix, first_n_shows_consider=None):
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
    true_positive_rate_vs_k = []
    no_first_100_shows_watched = np.count_nonzero(user_shows[19][:100])
    for i in range(*k):
        shows_sim_scores = find_top_k_shows(user_idx, i, recommend_matrix, 100)
        recommended_shows = [s[0] for s in shows_sim_scores]
        recommended_shows = np.array(recommended_shows)
        no_rec_shows_watched = np.count_nonzero(user_shows[19][recommended_shows])
        true_positive_rate_vs_k.append([i, (no_rec_shows_watched / no_first_100_shows_watched)])

    return np.array(true_positive_rate_vs_k)


def plot_true_positive_rate_vs_k(usr_true_positive_rate_vs_k, itm_true_positive_rate_vs_k):
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
    plt.show()
    plt.clf()


def kendall_rank_correlation(item_item, user_user, itemknn, wrmf):
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






