from sklearn.preprocessing import RobustScaler

class robust_scaler:
    def __init__(self):
        self.transformer = RobustScaler(copy=False)

    def __call__(self, supports):
        # supports have shape [num_users, num_data_points] or [num_data_points]
        return self.transformer.transform(supports)

    def train(self, data_points):
        self.transformer.fit(data_points)