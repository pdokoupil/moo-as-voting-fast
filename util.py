import numpy as np
from scipy.special import rel_entr

def kl_divergence(objective_weights, objectives):
    return sum(rel_entr(objective_weights, objectives))

# Returns list of per user kl divergences
# normalized hodnoty lze ziskat z toho pickle filu, r = pickle.load(...), r["normalized-per-user-mer"] etc.
def calculate_per_user_kl_divergence(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty, weights):
    
    print("Calculating KL-Divergence over normalized values")
    kl_divergences_normalized = [] #Over normalized objective values
    c = 0
    for mer, div, nov in zip(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty):
        objs = np.array([mer, div, nov])
        if np.any(objs <= 0):
            print(f"Warning: objs={objs} contains something non-positive")
            objs[objs <= 0] = 1e-6
            print(f"Replaced with epsilon: {objs}")
            c += 1
        normalized_objective_values = objs / objs.sum()

        assert np.all(np.isfinite(normalized_objective_values)), f"Normalized objective values must be finite: {normalized_objective_values}, {mer}, {div}, {nov}"
        divergence = kl_divergence(weights, normalized_objective_values)
        assert np.isfinite(divergence), f"KL-Divergence must be finite: {divergence}, {normalized_objective_values}, {mer}, {div}, {nov}"
        kl_divergences_normalized.append(divergence)
    

    return kl_divergences_normalized

# Returns per_user_mean_absolute_errors and per_user_errors
# per_user_mean_absolute_errors for each user is scalar value
# per_user_errors for each user is vector (length == # of objectives)
def calculate_per_user_errors(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty, weights):
    weights = np.array(weights)
    
    per_user_errors = []
    per_user_mean_absolute_errors = []
    for mer, div, nov in zip(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty):
        # Normalize to 1 sum
        objs = np.array([mer, div, nov])
        objs[objs <= 0] = 1e-6
        objs = objs / objs.sum()

        per_user_mean_absolute_errors.append(np.abs(objs - weights).mean())
        per_user_errors.append(objs - weights)

    return per_user_mean_absolute_errors, per_user_errors