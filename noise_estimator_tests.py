import numpy as np
import math
from gessaman.gessaman import Gessaman
from gessaman.utils import nn_estimator


if __name__ == '__main__':
    nb_simu = 10

    for i in range(nb_simu):
        n = 5000
        noise = 0.05

        X = np.array([[1 / n * i] for i in range(0, n)])
        Y = np.sqrt(X * (1 - X)) * np.sin((2 * math.pi * 1.05) / (X + .05)) + 0.5
        Y = Y.flatten()
        Y += np.random.normal(0, noise, n)

        alpha = 1/3
        g = Gessaman(alpha=alpha, nb_jobs=4, verbose=False)
        g.fit(X, Y)
        noise_estimators = [rule.std for rule in g.ruleset]
        gessaman_estimator = min(noise_estimators)

        neigh_estimator = nn_estimator.calc_1nn_noise_estimator(X, Y)

        print(f'Gessamen : {gessaman_estimator}')
        print(f'1NeighborsRegressor estimator: {neigh_estimator}')

        # Gessamen: 0.05059848114962786
        # 1NeighborsRegressorestimator: 0.001887572938761739
