import argparse
import os
import random
import time
from matplotlib.pyplot import axes

import numpy as np

from scipy.spatial.distance import squareform, pdist

from mandate_allocation.exactly_proportional_fuzzy_dhondt import exactly_proportional_fuzzy_dhondt
from mandate_allocation.exactly_proportional_fuzzy_dhondt_2 import exactly_proportional_fuzzy_dhondt_2
from mandate_allocation.fai_strategy import fai_strategy
from mandate_allocation.probabilistic_fai_strategy import probabilistic_fai_strategy
from mandate_allocation.weighted_average_strategy import weighted_average_strategy

from normalization.cdf import cdf
from normalization.standardization import standardization
from normalization.identity import identity
from normalization.robust_scaler import robust_scaler

from support.rating_based_relevance_support import rating_based_relevance_support
from support.intra_list_diversity_support import intra_list_diversity_support
from support.popularity_complement_support import popularity_complement_support

from mlflow import log_metric, log_param, log_artifacts, log_artifact, set_tracking_uri, set_experiment, start_run

from caserec.utils.process_data import ReadFile

from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction

def get_supports(users_partial_lists, items, extended_rating_matrix, distance_matrix, users_viewed_item, k):
    rel_supps = rating_based_relevance_support(extended_rating_matrix)
    div_supps = intra_list_diversity_support(users_partial_lists, items, distance_matrix, k)
    nov_supps = popularity_complement_support(users_viewed_item, num_users=users_partial_lists.shape[0])
    return np.stack([rel_supps, div_supps, nov_supps])

# Parse movielens metadata
def parse_metadata(metadata_path, item_to_item_id):
    metadata = dict()

    with open(metadata_path, encoding="ISO-8859-1") as f:
        all_genres = set()
        for line in f.readlines():
            [movie, movie_name, genres] = line.strip().split("::")
            genres = genres.split("|")
            all_genres.update(genres)
            metadata[int(movie)] = {
                "movie_name": movie_name,
                "genres": genres
            }
        
        genre_to_genre_id = {g:i for i, g in enumerate(all_genres)}
        metadata_matrix = np.zeros((len(item_to_item_id), len(all_genres)), dtype=np.int32)
        for movie, data in metadata.items():
            if movie not in item_to_item_id:
                continue
            item_id = item_to_item_id[movie]
            for g in data["genres"]:
                metadata_matrix[item_id, genre_to_genre_id[g]] = 1

    metadata_distances = np.float32(squareform(pdist(metadata_matrix, "cosine")))
    metadata_distances[np.isnan(metadata_distances)] = 1.0
    #metadata_matrix = 1.0 - metadata_matrix

    return metadata_distances

def get_baseline(args, baseline_factory):
    print(f"Getting baseline '{baseline_factory.__name__}'")
    baseline = baseline_factory(args.train_path)

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

    user_to_user_id = dict()
    user_id_to_user = dict()

    for idx, user in enumerate(train_set['users']):
        user_to_user_id[user] = idx
        user_id_to_user[idx] = user

    test_set = ReadFile(args.test_path).read()
    test_set_users = []

    test_set_users_start_index = 0
    next_user_idx = len(train_set['users'])
    for u in test_set['users']:
        if u not in user_to_user_id:
            print(f"Test set contains so-far-unknown user: {u}, assigning id: {next_user_idx}")
            if test_set_users_start_index == 0:
                test_set_users_start_index = next_user_idx
            user_to_user_id[u] = next_user_idx
            user_id_to_user[next_user_idx] = u
            user_estimated_rating = extended_rating_matrix.mean(axis=0, keepdims=True)
            extended_rating_matrix = np.concatenate([extended_rating_matrix, user_estimated_rating], axis=0)            
            unseen_items_mask = np.concatenate([unseen_items_mask, np.ones((1, num_items), dtype=np.bool8)])
            next_user_idx += 1

        test_set_users.append(user_to_user_id[u])
        
    metadata_distance_matrix = None
    if args.metadata_path:
        print("Parsing metadata")
        metadata_distance_matrix = parse_metadata(args.metadata_path, item_to_item_id)

    users = np.arange(extended_rating_matrix.shape[0]) # re-evaluate users because there can be new users in test set

    return items, users, \
        users_viewed_item, item_to_item_id, \
        item_id_to_item, extended_rating_matrix, \
        similarity_matrix, unseen_items_mask, \
        test_set_users_start_index, metadata_distance_matrix

def build_normalization(normalization_factory, shift):
    if shift:
        return normalization_factory(shift)
    else:
        return normalization_factory()

def prepare_normalization(normalization_factory, rating_matrix, distance_matrix, users_viewed_item, shift):
    num_users = rating_matrix.shape[0]

    relevance_data_points = rating_matrix.T
    print(relevance_data_points.mean(axis=1))
    
    upper_triangular_indices = np.triu_indices(distance_matrix.shape[0], k=1)
    upper_triangular_nonzero = distance_matrix[upper_triangular_indices]
        
    diversity_data_points = np.expand_dims(upper_triangular_nonzero, axis=1)
    novelty_data_points = np.expand_dims(1.0 - users_viewed_item / num_users, axis=1)

    norm_relevance = build_normalization(normalization_factory, shift)
    norm_relevance.train(relevance_data_points)
    
    norm_diversity = build_normalization(normalization_factory, shift)
    norm_diversity.train(diversity_data_points)

    norm_novelty = build_normalization(normalization_factory, shift)
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

    algorithm_factory = globals()[args.algorithm]
    print(f"Using '{args.algorithm}' algorithm")

    items, users, users_viewed_item, item_to_item_id, item_id_to_item, extended_rating_matrix, similarity_matrix, unseen_items_mask, test_set_users_start_index, metadata_distance_matrix = get_baseline(args, globals()[args.baseline])
    if args.diversity == "cb":
        print("Using content based diversity")
        distance_matrix = metadata_distance_matrix
    elif args.diversity == "cf":
        print("Using collaborative diversity")
        distance_matrix = 1.0 - similarity_matrix
    else:
        assert False, f"Unknown diversity: {args.diversity}"
    #extended_rating_matrix = (extended_rating_matrix - 1.0) / 4.0

    # Prepare normalizations
    start_time = time.perf_counter()
    normalizations = prepare_normalization(normalization_factory, extended_rating_matrix, distance_matrix, users_viewed_item, args.shift)
    print(f"Preparing normalizations took: {time.perf_counter() - start_time}")

    num_users = users.size
    users_partial_lists = np.full((num_users, args.k), -1, dtype=np.int32)
    
    obj_weights = args.weights
    obj_weights /= obj_weights.sum()
    mandate_allocation = algorithm_factory(obj_weights, args.masking_value)

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
    print(f"Lists: {users_partial_lists}")
    print(f"Item ID to Item: {item_id_to_item.items()}")
    #print(f"Mapped lists: {[item_id_to_item[item_id] for item_id in users_partial_lists[1]]}")
    #print(f"Lists ratings: {extended_rating_matrix[1, users_partial_lists[1]]}")

    custom_evaluate_voting(users_partial_lists, extended_rating_matrix, distance_matrix, users_viewed_item, normalizations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--train_path", type=str, default="/Users/pdokoupil/Downloads/filmtrust-folds/randomfilmtrustfolds/0/train.dat")
    parser.add_argument("--test_path", type=str, default="/Users/pdokoupil/Downloads/filmtrust-folds/randomfilmtrustfolds/0/test.dat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="0.3,0.3,0.3")
    parser.add_argument("--normalization", type=str, default="standardization")
    parser.add_argument("--algorithm", type=str, default="exactly_proportional_fuzzy_dhondt_2")
    parser.add_argument("--masking_value", type=float, default=-1e6)
    parser.add_argument("--baseline", type=str, default="MatrixFactorization")
    parser.add_argument("--metadata_path", type=str, default="/Users/pdokoupil/Downloads/ml-1m/movies.dat")
    parser.add_argument("--diversity", type=str, default="cf")
    parser.add_argument("--shift", type=float, default=None)
    args = parser.parse_args()

    args.weights = np.fromiter(map(float, args.weights.split(",")), dtype=np.float32)

    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)