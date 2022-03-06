import numpy as np

class probabilistic_fai_strategy:
    def __init__(self, obj_weights, *args):
        self.obj_weights = obj_weights

    # supports.shape[0] corresponds to number of objectives
    def __call__(self, masked_supports):
        curr_obj = np.random.choice(np.arange(masked_supports.shape[0]), p=self.obj_weights)
        return np.argmax(masked_supports[curr_obj], axis=1)