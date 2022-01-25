import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor


def calc_1nn_noise_estimator(X, Y):
    n = len(Y)
    if n < 2:
        return np.nan
    else:
        if n % 2 != 0:
            to_drop = random.sample(list(range(n)), 1)
            X = np.delete(X, to_drop)
            Y = np.delete(Y, to_drop)
            n -= 1

        neigh = KNeighborsRegressor(n_neighbors=1)
        sub_idx = random.sample(list(range(n)), int(n / 2))
        sub_idx_prime = list(filter(lambda v: v not in sub_idx, list(range(n))))
        sub_y = Y[sub_idx]
        sub_X = X[sub_idx]
        sub_y_prime = Y[sub_idx_prime]
        sub_X_prime = X[sub_idx_prime]

        if len(X.shape) == 1:
            neigh.fit(sub_X.reshape(-1, 1), sub_y)
            s_n = np.mean(neigh.predict(sub_X_prime.reshape(-1, 1)) * sub_y_prime)
        else:
            neigh.fit(sub_X, sub_y)
            s_n = np.mean(neigh.predict(sub_X_prime) * sub_y_prime)

        nn_estimator = np.mean(Y ** 2) - s_n

        return nn_estimator
