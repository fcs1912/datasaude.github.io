
import numpy as np
from collections import defaultdict
from surprise import Reader, Dataset, KNNBaseline

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
 
reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,55))
data = Dataset.load_from_file('base2.csv', reader=reader)
trainset = data.build_full_trainset()

#Train the algoritihm to compute the similarities between users
sim_options = {'name': 'pearson_baseline'}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()

predictions = algo.test(testset)

top_n = get_top_n(predictions, n=5)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print("{cidade:",uid,",doencas:", [iid for (iid, _) in user_ratings],"},")