import numpy as np
import math
import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gessaman.gessaman import Gessaman


if __name__ == "__main__":
    n = 5000
    noise = 0.05

    X = np.array([[1 / n * i] for i in range(0, n)])
    Y = np.sqrt(X * (1 - X)) * np.sin((2 * math.pi * 1.05) / (X + 0.05)) + 0.5
    Y = Y.flatten()
    Y += np.random.normal(0, noise, n)

    alpha_list = [i / 100 for i in list(range(50, 10, -1))]
    rectangles_list = []
    sigma_estimate = []
    for alpha in tqdm.tqdm(alpha_list):
        g = Gessaman(alpha=alpha, nb_jobs=4, verbose=False)
        g.fit(X, Y)
        noise_estimators = [rule.std for rule in g.ruleset]
        rule_estimator_index = noise_estimators.index(min(noise_estimators))
        rule_estimator = g.ruleset[rule_estimator_index]
        bmin = rule_estimator.condition.bmins[0]
        bmin = max(0, bmin)
        bmax = rule_estimator.condition.bmaxs[0]
        bmax = min(1.0, bmax)
        sigma_estimate.append(min(noise_estimators))
        rectangle = Rectangle(
            (bmin, np.min(Y)),
            bmax - bmin,
            np.max(Y) - np.min(Y),
            color="red",
            alpha=0.7,
        )
        rectangles_list.append(rectangle)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()

    for alpha, sigma, rectangle in zip(alpha_list, sigma_estimate, rectangles_list):
        ax.clear()
        ax.plot(X, Y)
        ax.add_patch(rectangle)
        ax.set_title(
            "alpha = " + str(round(alpha, 2)) + " & sigma_hat = " + str(round(sigma, 3))
        )
        fig.canvas.draw()
        time.sleep(0.5)

    time.sleep(2)
