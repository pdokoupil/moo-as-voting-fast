import numpy as np

class weighted_average_strategy:
    def __init__(self, obj_weights, *args):
        self.obj_weights = obj_weights[:, np.newaxis, np.newaxis]

    def __call__(self, masked_supports):
        return np.argmax(np.sum(masked_supports * self.obj_weights, axis=0), axis=1)