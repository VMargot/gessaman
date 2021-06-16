import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score

from gessaman.gessaman import Gessaman


def make_y(x, sigma, th_min=-0.4, th_max=0.4):
    y = [np.random.normal(-2, sigma) if x_val <= th_min else
         np.random.normal(0, sigma) if x_val <= th_max else
         np.random.normal(2, sigma) for x_val in x]
    return np.array(y)


def linear_data(X, noise):
    x_vect = X[:, 0]
    th_min = -0.4
    th_max = 0.4
    y = make_y(x_vect, noise, th_min, th_max)
    return X, y


def linear_data2(X, noise):
    x_vect = X[:, 0] + X[:, 1]
    th_min = -0.4
    th_max = 0.4

    y = make_y(x_vect, noise, th_min, th_max)
    return X, y


def circle_data(X, noise):
    x_vect = np.square(X[:, 0]) + np.square(X[:, 1])
    th_min = 0.5
    th_max = 0.8

    y = make_y(x_vect, noise, th_min, th_max)
    return X, y


h = 0.1  # step size in the mesh
n_samples = 1000
noise = 0.5

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    # "Naive Bayes",
    # "QDA",
    "Gessaman",
]

classifiers = [
    KNeighborsRegressor(3),
    SVR(kernel="linear", C=0.025),
    SVR(gamma=2, C=1),
    # GaussianProcessRegressor(1.0 * RBF(1.0)),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
    MLPRegressor(alpha=1),
    AdaBoostRegressor(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    Gessaman(nb_jobs=-1),
]

# X, y = make_classification(
#     n_features=2,
#     n_redundant=0,
#     n_informative=2,
#     random_state=1,
#     n_clusters_per_class=1,
#     n_samples=n_samples,
# )
# rng = np.random.RandomState(2)
# X += np.random.normal(0, noise, X.shape)
# linearly_separable = (X, y)

datasets = [
    linear_data,
    linear_data2,
    circle_data
]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, func in enumerate(datasets):
    nRows = 1000
    nCols = 2  # In this notebook, Y depends only on X1 and X2.
    # If nCols > 2 you add irrelevant columns to add noise (also interesting)
    noise = 1
    X = np.random.uniform(low=-1, high=1, size=(nRows, nCols))
    # preprocess dataset, split into training and test part
    X, y = func(X, noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print(name)
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        if hasattr(clf, "score"):
            score = clf.score(X_test, y_test)
        else:
            pred, _ = clf.predict(X_test, y_train)
            if name == 'Gessaman':
                print("sigma 2 estimation:", min([rule.std ** 2 for rule in clf.ruleset]))
                sub_y = np.extract(np.isfinite(pred), y_test)
                sub_pred = np.extract(np.isfinite(pred), pred)
                score = r2_score(sub_y, sub_pred)
            else:
                score = r2_score(y_test, pred)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        elif hasattr(clf, "predict_proba"):
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            if name == 'Gessaman':
                Z, _ = clf.predict(np.c_[xx.ravel(), yy.ravel()], y_train)
            else:
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            # Z = np.array([1.0 if z >= 0.5 else 0.0 for z in Z])

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", alpha=0.6)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.3)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - 0.01, yy.min() + 0.01, ("%.2f" % score).lstrip("0"), size=15, horizontalalignment="right")
        i += 1

plt.tight_layout()
plt.show()
