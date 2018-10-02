import collaborative_filtering as cf
from copy import deepcopy

k = [1, 20]
user_idx = 19

with open('user-shows.txt', 'r') as f:
    user_shows = cf.np.loadtxt(f)
    user_shows = user_shows.astype(int)

shows_list = []
with open('shows.txt', 'r') as f:
    for line in f:
        shows_list.append(line[1:len(line)-2])

shows = cf.np.array(shows_list)
del shows_list

user_shows_modified = deepcopy(user_shows)
user_shows_modified[19][:100] = 0

item_item_recommend_matrix = cf.item_item_recommend(user_shows_modified)
user_user_recommend_matrix = cf.user_user_recommend(user_shows_modified)

item_item_shows_sim_scores = cf.find_top_k_shows(user_idx, 5, item_item_recommend_matrix, 100)
user_user_shows_sim_scores = cf.find_top_k_shows(user_idx, 5, user_user_recommend_matrix, 100)

print("*** Item-Item Recommendation for User {0} with modified dataset: ***".format(user_idx))
for show_idx, sim_score in item_item_shows_sim_scores:
    print("{0} - {1}".format(shows[show_idx], sim_score))

print("\n*** User-User Recommendation for User {0} with modified dataset: ***".format(user_idx))
for show_idx, sim_score in user_user_shows_sim_scores:
    print("{0} - {1}".format(shows[show_idx], sim_score))

usr_true_positive_rate_vs_k = cf.find_true_positive_rate_vs_k(user_user_recommend_matrix,
                                                              user_shows, user_idx, k)
itm_true_positive_rate_vs_k = cf.find_true_positive_rate_vs_k(item_item_recommend_matrix,
                                                              user_shows, user_idx, k)

cf.plot_true_positive_rate_vs_k(usr_true_positive_rate_vs_k, itm_true_positive_rate_vs_k)

item_item_recommend_matrix = cf.item_item_recommend(user_shows)
user_user_recommend_matrix = cf.user_user_recommend(user_shows)
user_user_top10_shows_scores = cf.find_top_k_shows(user_idx, 10, user_user_recommend_matrix)
item_item_top10_shows_scores = cf.find_top_k_shows(user_idx, 10, item_item_recommend_matrix)

item_item = []
for show_idx, sim_score in item_item_top10_shows_scores:
    item_item.append(show_idx)

user_user = []
for show_idx, sim_score in user_user_top10_shows_scores:
    user_user.append(show_idx)

# MyMedialite Library Results
itemknn = [234, 48, 37, 543, 490, 477, 280, 553, 489, 222]
wrmf = [48,	77,	192, 208, 195,	280, 207, 222, 219,	489]

compare_ii_uu_itemknn_wrfm = [[item_item[i], user_user[i], itemknn[i], wrmf[i]] for i in range(10)]

print("\n*** Recommendation with various methods for user {0} with original dataset: ***".format(user_idx))
print("{0:^10s}{1:^10s}{2:^10s}{3:^10s}".format('Item-Item', 'User-User', 'ItemKNN', 'WRMF'))
for i in compare_ii_uu_itemknn_wrfm:
    for j in i:
        print("{0:^10d}".format(j), end='')
    print()

print()
all_recommend_show_ids = {*item_item, *user_user, *itemknn, *wrmf}
print("{0:^10s}{1:^s}".format('ShowID', 'ShowName'))
for i in sorted(all_recommend_show_ids):
    print("{0:^10d} - {1}".format(i, shows[i]))

cf.kendall_rank_correlation(item_item, user_user, itemknn, wrmf)

















