from gessaman.gessaman import Gessaman
from sklearn.datasets import load_boston, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import math

from gessaman.utils import nn_estimator


def make_y(x, noise, th_min=-0.4, th_max=0.4):
    y_vect = [-2 if x_val <= th_min else 0 if x_val <= th_max else 2 for x_val in x]
    y_vect += np.random.normal(0, noise, len(y_vect))
    return np.array(y_vect)


if __name__ == "__main__":
    # X, y = load_boston(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    #                                                     random_state=42)
    # g = Gessaman(nb_jobs=-1)
    # g.fit(X_train, y_train)
    # pred = g.predict(X_test)
    # print('Boston % of bad points: ', sum(1 - np.isfinite(pred)) / len(pred) * 100)
    # pred = np.nan_to_num(pred)
    # print('Boston: ', r2_score(y_test, pred))
    # print('Boston sigma 2:', min([rule.std**2 for rule in g.ruleset]))
    # # Boston:  0.44632846823132255

    # print("")
    # X, y = load_diabetes(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # g = Gessaman(alpha=1/3, nb_jobs=-1)
    # g.fit(X_train, y_train)
    # pred, bad_points = g.predict(X_test, y_train)
    # print("Diabetes % of bad points: ", sum(bad_points) / len(y_test))
    # pred = np.nan_to_num(pred)
    # print("Diabetes: ", r2_score(y_test, pred))
    # print("Diabetes EY2:", np.mean(y**2))
    # print("Diabetes sigma 2:", min([rule.std**2 for rule in g.ruleset]))
    # neigh_estimator = nn_estimator.calc_1nn_noise_estimator(X, y)
    # print("Diabetes 1NN sigma 2:", neigh_estimator)
    # # Diabetes:  0.2461021538303113
    # # Diabetes:  0.3686668958294672
    # # Diabetes sigma: 411.55102040816325

    noise = 0.05
    n = 5000
    alpha = None

    print("")
    print('Flag data simulation')

    nCols = 2
    h = 0.05

    np.random.seed(42)
    X = np.random.uniform(low=-1, high=1, size=(n, nCols))
    x_vect = X[:, 0]
    th_min = -0.4
    th_max = 0.4
    y = make_y(x_vect, noise, th_min, th_max)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    g = Gessaman(alpha=alpha, nb_jobs=-1)
    g.fit(X, y)
    # pred, bad_points = g.predict(X_test, y_train)
    # print("Simulation % of bad points: ", sum(bad_points) / len(y_test))
    # pred = np.nan_to_num(pred)
    # print("Simulation: ", r2_score(y_test, pred))
    # print("Simulation EY2:", np.mean(y ** 2))
    print(f'Mean of Y2: {np.mean(y ** 2)}')
    print(f'True sigma2: {noise ** 2}')
    print("Gessamen estimator:", min([rule.std**2 for rule in g.ruleset]))
    neigh_estimator = nn_estimator.calc_1nn_noise_estimator(X, y)
    print("1NeighborsRegressor estimator:", neigh_estimator)
    # Simulation: 0.6945311888168184
    # Simulation sigma: 0.7174451651251278

    print("")
    print('Simplified RuleFit data simulation')
    # Designing of training data
    nCols = 5

    X = np.random.randint(10, size=(n, nCols))
    X = X / 10.0

    y_true = (0.8 * np.exp(-2 * (X[:, 0] - X[:, 1]))
              + 2 * np.sin(math.pi * X[:, 2]) ** 2)
    y = y_true + np.random.normal(0, noise, n)

    g = Gessaman(alpha=alpha, nb_jobs=-1)
    g.fit(X, y)
    # pred, bad_points = g.predict(X_test, y_test)
    # print("Simulation % of bad points: ", sum(bad_points) / len(y_test))
    # pred = np.nan_to_num(pred)
    # print("Simulation: ", r2_score(y_test, pred))
    print(f'Mean of Y2: {np.mean(y ** 2)}')
    print(f'True sigma2: {noise ** 2}')
    print("Gessamen estimator:", min([rule.std**2 for rule in g.ruleset]))
    neigh_estimator = nn_estimator.calc_1nn_noise_estimator(X, y)
    print("1NeighborsRegressor estimator:", neigh_estimator)

    print('')
    print('Sigmoid data simulation')

    X = np.array([[1 / n * i + 0.1] for i in range(0, n)])
    # Y = np.sqrt(X * (1 - X)) * np.sin((2 * math.pi * 1.05) / (X + .05)) + 0.5
    Y = 1 / X * np.sin((2 * math.pi * 1.05) / (X + .05)) + 0.5
    # Y = np.sin((2 * math.pi * 1.05) / (X + .05)) + 0.5

    Y = Y.flatten()
    Y += np.random.normal(0, noise, n)

    g = Gessaman(alpha=alpha, nb_jobs=4, verbose=False)
    g.fit(X, Y)
    noise_estimators = [rule.std**2 for rule in g.ruleset]
    gessaman_estimator = min(noise_estimators)

    neigh_estimator = nn_estimator.calc_1nn_noise_estimator(X, Y)

    print(f'Mean of Y2: {np.mean(Y**2)}')
    print(f'True sigma2: {noise**2}')
    print(f'Gessamen estimator: {gessaman_estimator}')
    print(f'1NeighborsRegressor estimator: {neigh_estimator}')
