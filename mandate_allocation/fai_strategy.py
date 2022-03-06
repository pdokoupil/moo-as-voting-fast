import numpy as np

class fai_strategy:
    def __init__(self, *args):
        self.curr_obj = 1

    def __call__(self, masked_supports):
        res = np.argmax(masked_supports[self.curr_obj], axis=1)
        self.curr_obj = (self.curr_obj + 1) % masked_supports.shape[0]
        return res