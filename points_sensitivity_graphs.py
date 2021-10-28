import numpy as np
import pandas as pd
import seaborn as sns
import math
import tqdm
import matplotlib.pyplot as plt
from gessaman.gessaman import Gessaman
from gessaman.utils import nn_estimator
from sklearn.ensemble import RandomForestRegressor


def r2_score(y_test, predictions, y_hat):
    return 1 - np.mean((predictions - y_test)**2) / np.mean((y_test - y_hat)**2)


def data_sigmoid(nb_row, sigma2):
    x = np.array([[1 / n * i] for i in range(0, nb_row)])
    y = np.sqrt(x * (1 - x)) * np.sin((2 * math.pi * 1.05) / (x + 0.05)) + 0.5
    y = y.flatten()
    y += np.random.normal(0, math.sqrt(sigma2), nb_row)

    return x, y


def data_rulefit(nb_cols, nb_row, sigma2):
    x = np.random.randint(10, size=(nb_row, nb_cols))
    x = x / 10.0

    y_true = 0.8 * np.exp(-2 * (x[:, 0] - x[:, 1])) + 2 * np.sin(math.pi * x[:, 2]) ** 2
    y = y_true + np.random.normal(0, math.sqrt(sigma2), nb_row)

    return x, y


def data_flag(nb_cols, nb_row, sigma2):
    x = np.random.uniform(low=-1, high=1, size=(nb_row, nb_cols))
    th_min = -0.4
    th_max = 0.4
    y_true = [
        -2 if x_val <= th_min else 0 if x_val <= th_max else 2 for x_val in x[:, 0]
    ]
    y = y_true + np.random.normal(0, math.sqrt(sigma2), nb_row)

    return x, y


def data_diag(nb_cols, nb_row, sigma2):
    assert nb_cols >= 2, "You need at least 2 features for the diagonal simulations"
    x = np.random.uniform(low=-1, high=1, size=(nb_row, nb_cols))
    th_min = -0.4
    th_max = 0.4
    y_true = [
        -2
        if x_val[0] + x_val[1] <= th_min
        else 0
        if x_val[0] + x_val[1] <= th_max
        else 2
        for x_val in x
    ]
    y = y_true + np.random.normal(0, math.sqrt(sigma2), nb_row)

    return x, y


def data_circle(nb_cols, nb_row, sigma2):
    assert nb_cols >= 2, "You need at least 2 features for the diagonal simulations"
    x = np.random.uniform(low=-1, high=1, size=(nb_row, nb_cols))
    th_min = 0.5
    th_max = 0.8
    y_true = [
        -2
        if x_val[0] ** 2 + x_val[1] ** 2 <= th_min
        else 0
        if x_val[0] ** 2 + x_val[1] ** 2 <= th_max
        else 2
        for x_val in x
    ]
    y = y_true + np.random.normal(0, math.sqrt(sigma2), nb_row)

    return x, y


def do_graph(n, noise, step, nb_simu, data_type, d=1):
    min_points = int(n ** (1 / 2)) + 1
    k_list = list(range(min_points, int(n / 2), int(n * step)))
    sigma2_ratio_df = pd.DataFrame(index=list(range(nb_simu)), columns=k_list)
    nn_estimator_list = []
    r2_scores = pd.DataFrame(
        index=list(range(nb_simu * len(k_list) * 2)),
        columns=["% of points", "R2-score", "Algorithm"],
    )

    title = f"Points sensitivity for {data_type} dataset with sigma2={noise}, n={n} and for {nb_simu} simulations"
    j = 0
    for i in tqdm.tqdm(range(nb_simu)):
        if data_type == "sigmoid":
            x, y = data_sigmoid(n, noise)
        elif data_type == "rulefit":
            x, y = data_rulefit(d, n, noise)
        elif data_type == "flag":
            x, y = data_flag(d, n, noise)
        elif data_type == "diag":
            x, y = data_diag(d, n, noise)
        elif data_type == "circle":
            x, y = data_circle(d, n, noise)
        else:
            raise "Not implemented data_type"

        sigma2_estimate = []
        ymean = np.mean(y)
        for k in k_list:
            g = Gessaman(k=k, nb_jobs=4, verbose=False)
            g.fit(x, y)
            noise_estimators = [rule.std ** 2 for rule in g.ruleset]
            sigma2_estimate.append(min(noise_estimators))

            regr = RandomForestRegressor(min_samples_leaf=k)
            regr.fit(x, y)

            if data_type == "sigmoid":
                x_test, y_test = data_sigmoid(1000, noise)
            elif data_type == "rulefit":
                x_test, y_test = data_rulefit(d, 1000, noise)
            elif data_type == "flag":
                x_test, y_test = data_flag(d, 1000, noise)
            elif data_type == "diag":
                x_test, y_test = data_diag(d, 1000, noise)
            elif data_type == "circle":
                x_test, y_test = data_circle(d, 1000, noise)
            else:
                raise "Not implemented data_type"
            predictions, no_pred = g.predict(x_test, y)
            predictions = np.nan_to_num(predictions)
            r2_scores.iloc[j]["R2-score"] = r2_score(y_test, predictions, ymean)
            r2_scores.iloc[j]["% of points"] = round(k / n * 100, 2)
            r2_scores.iloc[j]["Algorithm"] = "Gessaman"
            j += 1

            predictions = regr.predict(x_test)
            r2_scores.iloc[j]["R2-score"] = r2_score(y_test, predictions, ymean)
            r2_scores.iloc[j]["% of points"] = round(k / n * 100, 2)
            r2_scores.iloc[j]["Algorithm"] = "RF"
            j += 1

        neigh_estimator = nn_estimator.calc_1nn_noise_estimator(x, y)
        nn_estimator_list.append(neigh_estimator)
        sigma2_ratio = [s / noise - 1 for s in sigma2_estimate]
        sigma2_ratio_df.iloc[i] = sigma2_ratio
        x_values = [k / n * 100 for k in k_list]
        plt.plot(x_values, sigma2_ratio, color="b", linestyle="--")

    x_values = [k / n * 100 for k in k_list]

    fig = plt.figure(figsize=(25, 17))
    plt.plot(
        x_values,
        sigma2_ratio_df.mean(0).values,
        color="black",
        label="Average Sigma2 Ratio",
        linestyle="-",
    )
    plt.axhline(
        y=np.mean(nn_estimator_list) / noise - 1, color="r", label="Average 1NN ratio"
    )
    plt.axhline(y=0, color="b", linestyle=":")

    plt.xlabel("% of points in each cell", fontsize=20)
    plt.ylabel("Sigma estimation ratio", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=20)
    plt.legend(loc="upper left", fontsize=15)
    # plt.show()
    fig.savefig(
        f"/home/vmargot/Documents/Jussieu/{data_type}_{n}_plot", format="svg", dpi=300
    )
    plt.close(fig)

    fig = plt.figure(figsize=(25, 17))
    plt.plot(
        sigma2_ratio_df.mean(0).values,
        color="black",
        label="Average Sigma2 Ratio",
        linestyle="-",
    )
    plt.axhline(
        y=np.mean(nn_estimator_list) / noise - 1, color="r", label="Average 1NN ratio"
    )
    plt.axhline(y=0, color="b", linestyle=":")
    sigma2_ratio_df.columns = [round(x, 1) for x in x_values]
    sns.boxplot(data=sigma2_ratio_df, dodge=False)

    plt.xlabel("% of points in each cell", fontsize=20)
    plt.ylabel("Sigma estimation ratio", fontsize=20)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=20)
    plt.legend(loc="upper left", fontsize=15)
    # plt.show()
    fig.savefig(
        f"/home/vmargot/Documents/Jussieu/{data_type}_{n}_boxplot",
        format="svg",
        dpi=300,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(25, 17))
    sns.boxplot(x="% of points", y="R2-score", hue="Algorithm", data=r2_scores)
    plt.xlabel("% of points in each cell", fontsize=20)
    plt.ylabel("R2 score", fontsize=20)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=20)
    plt.legend(loc="upper left", fontsize=15)
    # plt.show()
    fig.savefig(
        f"/home/vmargot/Documents/Jussieu/{data_type}_{n}_r2score",
        format="svg",
        dpi=300,
    )
    plt.close(fig)


if __name__ == "__main__":
    nb_simu = 50
    step = 0.02
    # The noise is chosen to have a signal / noise ratio > 2

    noise = 0.05
    n_list = [1125, 2500, 5000, 10000, 20000]
    for n in n_list:
        do_graph(n, noise, step, nb_simu, "sigmoid")

    d = 5
    noise = 0.3
    n_list = [1125, 2500, 5000, 10000, 20000]
    for n in n_list:
        do_graph(n, noise, step, nb_simu, "rulefit", d)

    d = 3
    noise = 1.0
    n_list = [1125, 2500, 5000, 10000, 20000]
    for n in n_list:
        do_graph(n, noise, step, nb_simu, "flag", d)

    d = 3
    noise = 1.2
    n_list = [1125, 2500, 5000, 10000, 20000]
    for n in n_list:
        do_graph(n, noise, step, nb_simu, "diag", d)

    d = 3
    noise = 1.5
    n_list = [1125, 2500, 5000, 10000, 20000]
    for n in n_list:
        do_graph(n, noise, step, nb_simu, "circle", d)
