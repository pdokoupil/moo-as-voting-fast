import argparse
import os
import random
import time
from matplotlib.pyplot import axes

import numpy as np

from normalization.cdf import cdf
from normalization.standardization import standardization
from normalization.identity import identity

from mlflow import log_metric, log_param, log_artifacts, log_artifact, set_tracking_uri, set_experiment, start_run

from caserec.utils.process_data import ReadFile

from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction

# Returns np.array with shape num_users x num_items which has meaning of support of item for user at step k
def relevance_supports(rating_matrix):
    return rating_matrix # simple

# users_viewed_item is np.array with length == #of items where each entry corresponds
# to the number of users who seen the particular item
def novelty_supports(users_viewed_item, num_users):
    return 1.0 - (users_viewed_item / num_users) # + proper expand_dims to ensure broadcasting

# old_obj_values are diversityy values for lists of length k - 1
def diversity_support(users_partial_lists, items, distance_matrix, k):
    # return ((old_obj_values * (k - 1)) + (distance_matrix[users_partial_list, item].sum() * 2)) / k

    # diversities between single top_k list an array of items (so per user)
    # distance_matrix[indices[:,np.newaxis], item_indices].sum(axis=0)

    # for multiple users we just work with rank 3 tensors .. results is 2d where first axis is user
    # expanding dims via newaxis is needed, alternative is to use d[rows][:,cols] syntax or np.ix_, however, both seems
    # to be slower
    #return ((old_obj_values * (k - 1)) + (distance_matrix[top_k_lists[:,:,np.newaxis], [1, 3]].sum(axis=1) * 2)) / k
    
    if k == 1:
        return np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), users_partial_lists.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
        #return np.repeat(distance_matrix[np.newaxis, items].sum(axis=1) / (distance_matrix.shape[0] - 1), users_partial_lists.shape[0], axis=0)
    elif k == 2:
        #old_supp = np.expand_dims(distance_matrix.sum(axis=1), axis=0) / (distance_matrix.shape[0] - 1)
        old_supp = distance_matrix[users_partial_lists[:, 0]].sum(axis=1, keepdims=True) / (distance_matrix.shape[0] - 1) # TODO remove, probably not necessary as it is constant per user
        #return distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].sum(axis=1) / (k - 1) - old_supp
        return 2 * distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].sum(axis=1) / k - old_supp
        
        # sups = np.zeros((users_partial_lists.shape[0], distance_matrix.shape[0]), dtype=np.float64)
        # for u in range(sups.shape[0]):
        #     old_supp = distance_matrix[users_partial_lists[u, 0]].sum() / (distance_matrix.shape[0] - 1)
        #     for i in items:
        #         user_top_k = np.concatenate([users_partial_lists[u, :k-1], [i]])
        #         sups[u, i] = distance_matrix[np.ix_(user_top_k, user_top_k)].sum() / k - old_supp
        # #return distance_matrix[users_partial_lists[:,:k-1,np.newaxis], users_partial_lists[:,:k-1]].sum() / k - old_supp
        # return sups
    return distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].sum(axis=1) / (k - 1)

def get_supports(users_partial_lists, items, extended_rating_matrix, distance_matrix, users_viewed_item, k):
    rel_supps = relevance_supports(extended_rating_matrix)
    #print(f"Diversity supports (shape={rel_supps.shape}): {rel_supps}")
    div_supps = diversity_support(users_partial_lists, items, distance_matrix, k)
    #print(f"Relevance supports (shape={div_supps.shape}): {div_supps}")
    nov_supps = np.repeat(novelty_supports(users_viewed_item, num_users=users_partial_lists.shape[0])[np.newaxis, :], users_partial_lists.shape[0], axis=0)
    #print(f"Novelty supports (shape={nov_supps.shape}): {nov_supps}")

    return np.stack([rel_supps, div_supps, nov_supps])

def get_baseline(args):
    baseline = MatrixFactorization(args.train_path)
    
    BaseRatingPrediction.compute(baseline)
    baseline.init_model()
    baseline.fit()
    baseline.create_matrix()
    similarity_matrix = baseline.compute_similarity(transpose=True)

    train_set = baseline.train_set
    
    num_items = len(train_set['items'])
    num_users = len(train_set['users'])

    unseen_items_mask = np.ones((num_users, num_items), dtype=np.bool8)
    unseen_items_mask[baseline.matrix > 0.0] = 0 # Mask out already seem items

    extended_rating_matrix = baseline.matrix.copy()
    for u_id in range(extended_rating_matrix.shape[0]):
        for i_id in range(extended_rating_matrix.shape[1]):
            if extended_rating_matrix[u_id, i_id] == 0.0:
                extended_rating_matrix[u_id, i_id] = baseline._predict_score(u_id, i_id)

    item_to_item_id = dict()
    item_id_to_item = dict()

    items = np.arange(num_items)
    users = np.arange(num_users)

    users_viewed_item = np.zeros_like(items, dtype=np.int32)

    for idx, item in enumerate(train_set['items']):
        item_to_item_id[item] = idx
        item_id_to_item[idx] = item
        users_viewed_item[idx] = len(train_set['users_viewed_item'][item])

    test_set = ReadFile(args.test_path).read()
    
    return items, users, users_viewed_item, item_to_item_id, item_id_to_item, extended_rating_matrix, similarity_matrix, unseen_items_mask
    

class batched_fai_strategy:
    def __init__(self):
        self.curr_obj = 1

    def __call__(self, masked_supports):
        res = np.argmax(masked_supports[self.curr_obj], axis=1)
        #self.curr_obj = (self.curr_obj + 1) % supports.shape[0]
        return res

class batched_weighted_average_strategy:
    def __init__(self, obj_weights):
        self.obj_weights = obj_weights[:, np.newaxis, np.newaxis]

    def __call__(self, masked_supports):
        return np.argmax(np.sum(masked_supports * self.obj_weights, axis=0), axis=1)

class batched_probabilistic_fai_strategy:
    def __init__(self, obj_weights):
        self.obj_weights = obj_weights

    # supports.shape[0] corresponds to number of objectives
    def __call__(self, masked_supports):
        curr_obj = np.random.choice(np.arange(masked_supports.shape[0]), p=self.obj_weights)
        return np.argmax(masked_supports[curr_obj], axis=1)

class batched_exactly_proportional_fuzzy_dhondt:
    def __init__(self, obj_weights):
        self.tot = None
        self.s_r = None
        self.votes = (obj_weights / obj_weights.sum())[:, np.newaxis, np.newaxis] # properly expand dims

    def __call__(self, mask, supports):
        masked_supports_neg = (mask * supports + (~mask) * args.masking_value)
        masked_supports = mask * supports
        if self.tot is None:
            # Shape should be [num_users, 1]
            self.tot = np.zeros((masked_supports.shape[1]), dtype=np.float64)
            # Shape is [num_parties, num_users]
            self.s_r = np.zeros((masked_supports.shape[0], masked_supports.shape[1]), dtype=np.float64)

        # shape [num_users, num_items]
        tot_items = np.full((1, masked_supports.shape[1], masked_supports.shape[2]), self.tot[:, np.newaxis])
        tot_items += np.sum(masked_supports, axis=0)

        # Shape of e_r should be [num_objs, num_users, num_items]
        e_r = np.maximum(0.0, tot_items * self.votes - self.s_r[..., np.newaxis])
        # Shape should be [num_users, num_items]
        gain_items = np.minimum(masked_supports_neg, e_r).sum(axis=0) #np.minimum(masked_supports, e_r).sum(axis=0)

        # Shape should be [num_users,]
        max_gain_items = np.argmax(gain_items, axis=1)

        self.s_r += np.squeeze(np.take_along_axis(masked_supports, max_gain_items[np.newaxis, :, np.newaxis], axis=2))
        self.tot = self.s_r.sum(axis=0)

        return max_gain_items

class batched_exactly_proportional_fuzzy_dhondt_2:
    def __init__(self, obj_weights):
        self.tot = None
        self.s_r = None
        self.votes = (obj_weights / obj_weights.sum())[:, np.newaxis, np.newaxis] # properly expand dims

    def __call__(self, mask, supports):
        masked_supports_neg = (mask * supports + (~mask) * args.masking_value)
        masked_supports = mask * supports
        if self.tot is None:
            # Shape should be [num_users, 1]
            self.tot = np.zeros((masked_supports.shape[1]), dtype=np.float64)
            # Shape is [num_parties, num_users]
            self.s_r = np.zeros((masked_supports.shape[0], masked_supports.shape[1]), dtype=np.float64)

        # shape [num_users, num_items]
        tot_items = np.full((1, masked_supports.shape[1], masked_supports.shape[2]), self.tot[:, np.newaxis])
        tot_items += np.sum(np.maximum(0.0, masked_supports), axis=0)

        # Shape of e_r should be [num_objs, num_users, num_items]
        unused_p = tot_items * self.votes - self.s_r[..., np.newaxis]
        
        positive_support_mask = masked_supports_neg >= 0.0
        negative_support_mask = masked_supports_neg < 0.0
        gain_items = np.zeros_like(masked_supports, dtype=np.float64)
        gain_items[positive_support_mask] = np.maximum(0, np.minimum(masked_supports_neg[positive_support_mask], unused_p[positive_support_mask]))
        #np.put(gain_items, positive_support_mask, np.maximum(0, np.minimum(np.take(masked_supports, positive_support_mask), unused_p)))
        gain_items[negative_support_mask] = np.minimum(0, masked_supports_neg[negative_support_mask] - unused_p[negative_support_mask])
        #np.put(gain_items, negative_support_mask, np.minimum(0, np.take(masked_supports, negative_support_mask) - unused_p))

        # Shape should be [num_users, num_items]
        gain_items = gain_items.sum(axis=0)

        # Shape should be [num_users,]
        #max_gain_items = np.argmax(gain_items, axis=1)
        max_gains = np.max(gain_items, axis=1, keepdims=True)
        #all_max_gain_items = np.where(gain_items < max_gains)
        # Break ties by selecting items with maximal tot_items
        max_gain_items = np.argmax(np.where(gain_items == max_gains, np.squeeze(tot_items), np.NINF), axis=1)

        self.s_r += np.squeeze(np.take_along_axis(masked_supports, max_gain_items[np.newaxis, :, np.newaxis], axis=2))
        self.tot = np.where(self.s_r >= 0, self.s_r, 0).sum(axis=0) # TODO NEW
        # self.tot = self.s_r.sum(axis=0)

        return max_gain_items

def prepare_normalization(normalization_factory, rating_matrix, distance_matrix, users_viewed_item):
    num_users = rating_matrix.shape[0]

    relevance_data_points = rating_matrix.T
    
    upper_triangular_indices = np.triu_indices(distance_matrix.shape[0], k=1)
    upper_triangular_nonzero = distance_matrix[upper_triangular_indices]
        
    diversity_data_points = np.expand_dims(upper_triangular_nonzero, axis=1)
    novelty_data_points = np.expand_dims(1.0 - users_viewed_item / num_users, axis=1)

    norm_relevance = normalization_factory()
    norm_relevance.train(relevance_data_points)
    
    norm_diversity = normalization_factory()
    norm_diversity.train(diversity_data_points)

    norm_novelty = normalization_factory()
    norm_novelty.train(novelty_data_points)

    return [norm_relevance, norm_diversity, norm_novelty]

def custom_evaluate_voting(top_k, rating_matrix, distance_matrix, users_viewed_item, normalizations):
    total_mer = 0.0
    total_novelty = 0.0
    total_diversity = 0.0
    n = 0

    per_user_mer = []
    per_user_diversity = []
    per_user_novelty = []

    for user_id, user_ranking in enumerate(top_k):
        
        relevance = rating_matrix[user_id][user_ranking].sum()
        novelty = (1.0 - users_viewed_item[user_ranking] / rating_matrix.shape[0]).sum()
        diversity = distance_matrix[np.ix_(user_ranking, user_ranking)].sum() / user_ranking.size

        per_user_mer.append(relevance)
        per_user_diversity.append(diversity)
        per_user_novelty.append(novelty)

        total_mer += relevance
        total_diversity += diversity
        total_novelty += novelty
        n += 1
        
    total_mer = total_mer / n
    total_diversity = total_diversity / n
    total_novelty = total_novelty / n

    print(f"MEAN ESTIMATED RATING: {total_mer}")
    print(f"DIVERSITY2: {total_diversity}")
    print(f"NOVELTY2: {total_novelty}")
    print("-------------------")
    log_metric("raw_mer", total_mer)
    log_metric("raw_diversity", total_diversity)
    log_metric("raw_novelty", total_novelty)

    
    [mer_norm, div_norm, nov_norm] = normalizations

    num_users = top_k.shape[0]    

    normalized_mer = 0.0
    normalized_diversity = 0.0
    normalized_novelty = 0.0
    
    normalized_per_user_mer = []
    normalized_per_user_diversity = []
    normalized_per_user_novelty = []

    normalized_per_user_mer_matrix = mer_norm(np.mean(np.take_along_axis(rating_matrix, top_k, axis=1), axis=1, keepdims=True).T).T

    # Calculate normalized MER per user
    n = 0
    for user_id, user_ranking in enumerate(top_k):
        
        relevance = rating_matrix[user_id][user_ranking].sum()
        novelty = (1.0 - users_viewed_item[user_ranking] / rating_matrix.shape[0]).sum()
        diversity = distance_matrix[np.ix_(user_ranking, user_ranking)].sum() / user_ranking.size

        #normalized_per_user_mer.append(mer_norm[user_id](rating_matrix[user_id, user_ranking].mean().reshape(-1, 1)))
        normalized_per_user_mer.append(normalized_per_user_mer_matrix[user_id])
        normalized_mer += normalized_per_user_mer[-1]

        #normalized_per_user_diversity.append(div_norm((distance_matrix[np.ix_(user_ranking, user_ranking)].sum() / 2).mean().reshape(-1, 1)))
        
        
        upper_triangular = np.triu(distance_matrix[np.ix_(user_ranking, user_ranking)], k=1)
        upper_triangular_nonzero_mean = upper_triangular.sum() / ((upper_triangular.size - upper_triangular.shape[0]) / 2)
        normalized_per_user_diversity.append(div_norm(upper_triangular_nonzero_mean.reshape(-1, 1)))
        normalized_diversity += normalized_per_user_diversity[-1]

        normalized_per_user_novelty.append(nov_norm((1.0 - users_viewed_item[user_ranking] / num_users).mean().reshape(-1, 1)))
        normalized_novelty += normalized_per_user_novelty[-1]

        per_user_mer.append(relevance)
        per_user_diversity.append(diversity)
        per_user_novelty.append(novelty)

        total_mer += relevance
        total_diversity += diversity
        total_novelty += novelty
        n += 1


    normalized_mer = normalized_mer.item() / n
    normalized_diversity = normalized_diversity.item() / n
    normalized_novelty = normalized_novelty.item() / n

    print(f"Normalized MER: {normalized_mer}")
    print(f"Normalized DIVERSITY2: {normalized_diversity}")
    print(f"Normalized NOVELTY2: {normalized_novelty}")
    log_metric("normalized_mer", normalized_mer)
    log_metric("normalized_diversity", normalized_diversity)
    log_metric("normalized_novelty", normalized_novelty)

    # Print sum-to-1 results
    s = normalized_mer + normalized_diversity + normalized_novelty
    print(f"Sum-To-1 Normalized MER: {normalized_mer / s}")
    print(f"Sum-To-1 Normalized DIVERSITY2: {normalized_diversity / s}")
    print(f"Sum-To-1 Normalized NOVELTY2: {normalized_novelty / s}")
    log_metric("normalized_sum_to_one_mer", normalized_mer / s)
    log_metric("normalized_sum_to_one_diversity", normalized_diversity / s)
    log_metric("normalized_sum_to_one_novelty", normalized_novelty / s)
    
def main(args):
    for arg_name in dir(args):
        if arg_name[0] != '_':
            arg_value = getattr(args, arg_name)
            print(f"\t{arg_name}={arg_value}")

    if not args.normalization:
        print(f"Using Identity normalization")
        normalization_factory = identity
    else:
        print(f"Using {args.normalization} normalization")
        normalization_factory = globals()[args.normalization]

    algorithm_factory = globals()[f"batched_{args.algorithm}"]
    print(f"Using 'batched_{args.algorithm}' algorithm")

    items, users, users_viewed_item, item_to_item_id, item_id_to_item, extended_rating_matrix, similarity_matrix, unseen_items_mask = get_baseline(args)
    distance_matrix = 1.0 - similarity_matrix
    #extended_rating_matrix = (extended_rating_matrix - 1.0) / 4.0

    # Prepare normalizations
    start_time = time.perf_counter()
    normalizations = prepare_normalization(normalization_factory, extended_rating_matrix, distance_matrix, users_viewed_item)
    print(f"Preparing normalizations took: {time.perf_counter() - start_time}")

    num_users = users.size
    users_partial_lists = np.full((num_users, args.k), -1, dtype=np.int32)
    
    obj_weights = args.weights
    obj_weights /= obj_weights.sum()
    mandate_allocation = algorithm_factory(obj_weights)

    start_time = time.perf_counter()

    # Masking already recommended users and SEEN items
    mask = unseen_items_mask.copy()
    
    for i in range(args.k):
        # Calculate support values
        supports = get_supports(users_partial_lists, items, extended_rating_matrix, distance_matrix, users_viewed_item, k=i+1)
        print(f"Diversity supports: {supports[1, 0, :5]}")
        # Normalize the supports
        assert supports.shape[0] == 3, "expecting 3 objectives, if updated, update code below"
        
        supports[0, :, :] = normalizations[0](supports[0].T).T
        supports[1, :, :] = normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1))
        supports[2, :, :] = normalizations[2](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1))
        
        # Mask out the already recommended items
        np.put_along_axis(mask, users_partial_lists[:, :i], 0, 1)

        # Masked supports
        masked_supports = (mask * supports + (~mask) * args.masking_value)
        print(f"Diversity supports MASKED: {masked_supports[1, 0, :5]}")
        
        # Get the per-user top-k recommendations
        users_partial_lists[:, i] = mandate_allocation(mask, supports)

    print(f"### Whole prediction took: {time.perf_counter() - start_time} ###")
    print(f"Lists: {users_partial_lists[1]}")
    print(f"Mapped lists: {[item_id_to_item[item_id] for item_id in users_partial_lists[1]]}")
    print(f"Lists ratings: {extended_rating_matrix[1, users_partial_lists[1]]}")
    custom_evaluate_voting(users_partial_lists, extended_rating_matrix, distance_matrix, users_viewed_item, normalizations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--train_path", type=str, default="/Users/pdokoupil/Downloads/ml-100k-folds/newlightfmfolds/0/train.dat")
    parser.add_argument("--test_path", type=str, default="/Users/pdokoupil/Downloads/ml-100k-folds/newlightfmfolds/0/test.dat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="0.3,0.3,0.3")
    parser.add_argument("--normalization", type=str, default="standardization")
    parser.add_argument("--algorithm", type=str, default="exactly_proportional_fuzzy_dhondt_2")
    parser.add_argument("--masking_value", type=float, default=-1e6)
    args = parser.parse_args()

    args.weights = np.fromiter(map(float, args.weights.split(",")), dtype=np.float32)

    np.random.seed(args.seed)
    random.seed(args.seed)

    #set_tracking_uri("http://gpulab.ms.mff.cuni.cz:7022")
    #set_experiment("moo-as-voting-fast")
    
    #with start_run():
    main(args)


