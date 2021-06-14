from gessaman.gessaman import Gessaman
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


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

    print("")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    g = Gessaman(nb_jobs=-1)
    g.fit(X_train, y_train)
    pred, bad_points = g.predict(X_test, y_train)
    print("Diabetes % of bad points: ", sum(bad_points) / len(y_test))
    pred = np.nan_to_num(pred)
    print("Diabetes: ", r2_score(y_test, pred))
    print("Diabetes sigma 2:", min([rule.std ** 2 for rule in g.ruleset]))
    # Diabetes:  0.2461021538303113
    # Diabetes:  0.3686668958294672
    # Diabetes sigma 2: 411.55102040816325

    print("")
    nRows = 5000
    nCols = 2
    noise = 1.0
    h = 0.05

    np.random.seed(42)
    X = np.random.uniform(low=-1, high=1, size=(nRows, nCols))
    x_vect = X[:, 0]
    th_min = -0.4
    th_max = 0.4
    y = make_y(x_vect, noise, th_min, th_max)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    g = Gessaman(nb_jobs=-1)
    g.fit(X_train, y_train)
    pred, bad_points = g.predict(X_test, y_train)
    print("Simulation % of bad points: ", sum(bad_points) / len(y_test))
    pred = np.nan_to_num(pred)
    print("Simulation: ", r2_score(y_test, pred))
    print("Simulation sigma 2:", min([rule.std ** 2 for rule in g.ruleset]))
    # Simulation: 0.6945311888168184
    # Simulation sigma2: 0.7174451651251278
