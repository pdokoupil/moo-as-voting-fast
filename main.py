import argparse
import os
import pickle
import random
import time

import mlflow

import numpy as np

from scipy.spatial.distance import squareform, pdist

from util import calculate_per_user_kl_divergence, calculate_per_user_errors

from mandate_allocation.exactly_proportional_fuzzy_dhondt import exactly_proportional_fuzzy_dhondt
from mandate_allocation.exactly_proportional_fuzzy_dhondt_2 import exactly_proportional_fuzzy_dhondt_2
from mandate_allocation.fai_strategy import fai_strategy
from mandate_allocation.probabilistic_fai_strategy import probabilistic_fai_strategy
from mandate_allocation.weighted_average_strategy import weighted_average_strategy
from mandate_allocation.sainte_lague_method import sainte_lague_method

from normalization.cdf import cdf
from normalization.standardization import standardization
from normalization.identity import identity
from normalization.robust_scaler import robust_scaler
from normalization.cdf_threshold_shift import cdf_threshold_shift

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

def save_cache(cache_path, cache):
    print(f"Saving cache to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

def load_cache(cache_path):
    print(f"Loading cache from: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    return cache

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
    
    cache_path = os.path.join(args.cache_dir, f"baseline_{baseline_factory.__name__}_{args.seed}.pckl")
    if args.cache_dir and os.path.exists(cache_path):
        cache = load_cache(cache_path)
        items = cache["items"]    
        users = cache["users"]
        users_viewed_item = cache["users_viewed_item"]
        item_to_item_id = cache["item_to_item_id"]
        item_id_to_item = cache["item_id_to_item"]
        extended_rating_matrix = cache["extended_rating_matrix"]
        similarity_matrix = cache["similarity_matrix"]
        unseen_items_mask = cache["unseen_items_mask"]
        test_set_users_start_index = cache["test_set_users_start_index"]
    else:
        print(f"Calculating baseline '{baseline_factory.__name__}'")
        baseline = baseline_factory(args.train_path)

        BaseRatingPrediction.compute(baseline)
        baseline.init_model()
        if hasattr(baseline, "fit"):
            baseline.fit()
        elif hasattr(baseline, "train_baselines"):
            baseline.train_baselines()
        else:
            assert False, "Fit/train_baselines not found for baseline"
        baseline.create_matrix()
        similarity_matrix = baseline.compute_similarity(transpose=True)

        train_set = baseline.train_set

        num_items = len(train_set['items'])
        num_users = len(train_set['users'])

        unseen_items_mask = np.ones((num_users, num_items), dtype=np.bool8)
        unseen_items_mask[baseline.matrix > 0.0] = 0 # Mask out already seem items
        
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
        
        if baseline_factory == ItemKNN:
            print("Injecting into ItemKNN")
            def predict_score_wrapper(u_id, i_id):
                res = baseline.predict_scores(user_id_to_user[u_id], [item_id_to_item[i_id]])
                print(f"Predicting: {res}")
                return res
            setattr(baseline, "_predict_score", predict_score_wrapper)

        extended_rating_matrix = baseline.matrix.copy()
        for u_id in range(extended_rating_matrix.shape[0]):
            for i_id in range(extended_rating_matrix.shape[1]):
                if extended_rating_matrix[u_id, i_id] == 0.0:
                    extended_rating_matrix[u_id, i_id] = baseline._predict_score(u_id, i_id)

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
            
        users = np.arange(extended_rating_matrix.shape[0]) # re-evaluate users because there can be new users in test set

        if args.cache_dir:
            cache = {
                "items": items,
                "users": users,
                "users_viewed_item": users_viewed_item,
                "item_to_item_id": item_to_item_id,
                "item_id_to_item": item_id_to_item,
                "extended_rating_matrix": extended_rating_matrix,
                "similarity_matrix": similarity_matrix,
                "unseen_items_mask": unseen_items_mask,
                "test_set_users_start_index": test_set_users_start_index
            }
            save_cache(cache_path, cache)

    metadata_distance_matrix = None
    if args.metadata_path:
        print(f"Parsing metadata from path: '{args.metadata_path}'")
        metadata_distance_matrix = parse_metadata(args.metadata_path, item_to_item_id)

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

def prepare_normalization(args, normalization_factory, rating_matrix, distance_matrix, users_viewed_item, shift):
    cache_path = os.path.join(args.cache_dir, f"sup_norm_{normalization_factory.__name__}_{shift}_{args.seed}.pckl")
    if args.cache_dir and os.path.exists(cache_path):
        cache = load_cache(cache_path)
        norm_relevance = cache["norm_relevance"]
        norm_diversity = cache["norm_diversity"]
        norm_novelty = cache["norm_novelty"]
    else:
        num_users = rating_matrix.shape[0]

        relevance_data_points = rating_matrix.T
        
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

        if args.cache_dir:
            cache = {
                "norm_relevance": norm_relevance,
                "norm_diversity": norm_diversity,
                "norm_novelty": norm_novelty
            }
            save_cache(cache_path, cache)

    return [norm_relevance, norm_diversity, norm_novelty]

def custom_evaluate_voting(top_k, rating_matrix, distance_matrix, users_viewed_item, normalizations, obj_weights, discount_sequences):
    start_time = time.perf_counter()
    
    [mer_norm, div_norm, nov_norm] = normalizations

    num_users = top_k.shape[0]    

    normalized_mer = 0.0
    normalized_diversity = 0.0
    normalized_novelty = 0.0
    
    normalized_per_user_mer = []
    normalized_per_user_diversity = []
    normalized_per_user_novelty = []

    normalized_per_user_mer_matrix = mer_norm(np.sum(np.take_along_axis(rating_matrix, top_k, axis=1) * discount_sequences[0], axis=1, keepdims=True).T / discount_sequences[0].sum(), ignore_shift=False).T
    
    total_mer = 0.0
    total_novelty = 0.0
    total_diversity = 0.0
    
    per_user_mer = []
    per_user_diversity = []
    per_user_novelty = []
    n = 0
    for user_id, user_ranking in enumerate(top_k):
        
        relevance = (rating_matrix[user_id][user_ranking] * discount_sequences[0]).sum()
        novelty = ((1.0 - users_viewed_item[user_ranking] / rating_matrix.shape[0]) * discount_sequences[2]).sum()
        div_discount = np.repeat(np.expand_dims(discount_sequences[1], axis=0).T, user_ranking.size, axis=1)
        diversity = (distance_matrix[np.ix_(user_ranking, user_ranking)] * div_discount).sum() / user_ranking.size

        # Per user MER
        normalized_per_user_mer.append(normalized_per_user_mer_matrix[user_id].item())
        normalized_mer += normalized_per_user_mer[-1]
        
        # Per user Diversity
        ranking_distances = distance_matrix[np.ix_(user_ranking, user_ranking)] * div_discount
        triu_indices = np.triu_indices(user_ranking.size, k=1)
        ranking_distances_mean = ranking_distances[triu_indices].sum() / div_discount[triu_indices].sum()
        normalized_ranking_distances_mean = div_norm([[ranking_distances_mean]], ignore_shift=False)
        normalized_per_user_diversity.append(normalized_ranking_distances_mean.item())
        normalized_diversity += normalized_per_user_diversity[-1]

        # Per user novelty
        normalized_per_user_novelty.append(nov_norm(((1.0 - users_viewed_item[user_ranking] / num_users) * discount_sequences[2]).sum().reshape(-1, 1) / discount_sequences[2].sum(), ignore_shift=False).item())
        normalized_novelty += normalized_per_user_novelty[-1]
        
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

    normalized_mer = normalized_mer / n
    normalized_diversity = normalized_diversity / n
    normalized_novelty = normalized_novelty / n

    per_user_kl_divergence = calculate_per_user_kl_divergence(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty, obj_weights)
    per_user_mean_absolute_errors, per_user_errors = calculate_per_user_errors(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty, obj_weights)

    print(f"per_user_kl_divergence: {per_user_kl_divergence}")
    print(f"per_user_mean_absolute_errors: {per_user_mean_absolute_errors}")
    print(f"per_user_errors: {per_user_errors}")

    print("####################")
    print(f"MEAN ESTIMATED RATING: {total_mer}")
    print(f"DIVERSITY2: {total_diversity}")
    print(f"NOVELTY2: {total_novelty}")
    print("--------------------")
    log_metric("raw_mer", total_mer)
    log_metric("raw_diversity", total_diversity)
    log_metric("raw_novelty", total_novelty)

    print(f"Normalized MER: {normalized_mer}")
    print(f"Normalized DIVERSITY2: {normalized_diversity}")
    print(f"Normalized NOVELTY2: {normalized_novelty}")
    print("--------------------")
    log_metric("normalized_mer", normalized_mer)
    log_metric("normalized_diversity", normalized_diversity)
    log_metric("normalized_novelty", normalized_novelty)

    # Print sum-to-1 results
    s = normalized_mer + normalized_diversity + normalized_novelty
    print(f"Sum-To-1 Normalized MER: {normalized_mer / s}")
    print(f"Sum-To-1 Normalized DIVERSITY2: {normalized_diversity / s}")
    print(f"Sum-To-1 Normalized NOVELTY2: {normalized_novelty / s}")
    print("--------------------")
    log_metric("normalized_sum_to_one_mer", normalized_mer / s)
    log_metric("normalized_sum_to_one_diversity", normalized_diversity / s)
    log_metric("normalized_sum_to_one_novelty", normalized_novelty / s)

    mean_kl_divergence = np.mean(per_user_kl_divergence)
    mean_absolute_error = np.mean(per_user_mean_absolute_errors)
    mean_error = np.mean(per_user_errors)

    print(f"mean_kl_divergence: {mean_kl_divergence}")
    print(f"mean_absolute_error: {mean_absolute_error}")
    print(f"mean_error: {mean_error}")
    print("####################")
    log_metric("mean_kl_divergence", mean_kl_divergence)
    log_metric("mean_absolute_error", mean_absolute_error)
    log_metric("mean_error", mean_error)

    print(f"Evaluation took: {time.perf_counter() - start_time}")

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
        assert args.metadata_path, "Metadata path must be specified when using cb diversity"
        distance_matrix = metadata_distance_matrix
    elif args.diversity == "cf":
        print("Using collaborative diversity")
        distance_matrix = 1.0 - similarity_matrix
    else:
        assert False, f"Unknown diversity: {args.diversity}"
    #extended_rating_matrix = (extended_rating_matrix - 1.0) / 4.0

    # Prepare normalizations
    start_time = time.perf_counter()
    normalizations = prepare_normalization(args, normalization_factory, extended_rating_matrix, distance_matrix, users_viewed_item, args.shift)
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
        iter_start_time = time.perf_counter()
        print(f"Predicting for i: {i + 1} out of: {args.k}")
        # Calculate support values
        supports = get_supports(users_partial_lists, items, extended_rating_matrix, distance_matrix, users_viewed_item, k=i+1)
        
        # Normalize the supports
        assert supports.shape[0] == 3, "expecting 3 objectives, if updated, update code below"
        
        supports[0, :, :] = normalizations[0](supports[0].T).T * args.discount_sequences[0][i]
        supports[1, :, :] = normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * args.discount_sequences[1][i]
        supports[2, :, :] = normalizations[2](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1)) * args.discount_sequences[2][i]
        
        # Mask out the already recommended items
        np.put_along_axis(mask, users_partial_lists[:, :i], 0, 1)

        # Get the per-user top-k recommendations
        users_partial_lists[:, i] = mandate_allocation(mask, supports)

        print(f"i: {i + 1} done, took: {time.perf_counter() - iter_start_time}")

    print(f"### Whole prediction took: {time.perf_counter() - start_time} ###")
    print(f"Lists: {users_partial_lists.tolist()}")
    print(f"Item ID to Item: {item_id_to_item.items()}")
    mapped_lists = np.fromiter(map(item_id_to_item.__getitem__, users_partial_lists.flatten()), dtype=np.int32).reshape(users_partial_lists.shape).tolist()
    print(f"Mapped lists: {mapped_lists}")

    custom_evaluate_voting(users_partial_lists, extended_rating_matrix, distance_matrix, users_viewed_item, normalizations, obj_weights, args.discount_sequences)

    if args.artifact_dir:
        log_artifacts(args.artifact_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--train_path", type=str, default="/Users/pdokoupil/Downloads/filmtrust-folds/randomfilmtrustfolds/0/train.dat")
    parser.add_argument("--test_path", type=str, default="/Users/pdokoupil/Downloads/filmtrust-folds/randomfilmtrustfolds/0/test.dat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="0.3,0.3,0.3")
    parser.add_argument("--normalization", type=str, default="cdf_threshold_shift")
    parser.add_argument("--algorithm", type=str, default="exactly_proportional_fuzzy_dhondt_2")
    parser.add_argument("--masking_value", type=float, default=-1e6)
    parser.add_argument("--baseline", type=str, default="MatrixFactorization")
    parser.add_argument("--metadata_path", type=str, default="/Users/pdokoupil/Downloads/ml-1m/movies.dat")
    parser.add_argument("--diversity", type=str, default="cf")
    parser.add_argument("--shift", type=float, default=-0.1)
    parser.add_argument("--cache_dir", type=str, default=".")
    parser.add_argument("--artifact_dir", type=str, default=None)
    parser.add_argument("--output_path_prefix", type=str, default=None)
    parser.add_argument("--discounts", type=str, default="1,1,1")
    args = parser.parse_args()

    args.weights = np.fromiter(map(float, args.weights.split(",")), dtype=np.float32)
    args.discounts = [float(d) for d in args.discounts.split(",")]
    args.discount_sequences = np.stack([np.geomspace(start=1.0,stop=d**args.k, num=args.k, endpoint=False) for d in args.discounts], axis=0)

    if not args.artifact_dir:
        print("Artifact directory is not specified, trying to set it")
        run_id = os.environ[mlflow.tracking._RUN_ID_ENV_VAR] if mlflow.tracking._RUN_ID_ENV_VAR in os.environ else None
        if not run_id:
            print("Not inside mlflow's run, leaving artifact directory empty")
        else:
            print(f"Inside mlflow's run {run_id} setting artifact directory")
            if args.output_path_prefix:
                args.artifact_dir = os.path.join(args.output_path_prefix, run_id)
                print(f"Set artifact directory to: {args.artifact_dir}")
            else:
                print("Output path prefix is not set, skipping setting of artifact directory")

    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)