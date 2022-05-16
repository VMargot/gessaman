import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro, kstest
import math
from scipy.stats import chi2, norm
import tqdm
import matplotlib.pyplot as plt
from gessaman.gessaman import Gessaman
from gessaman.utils import nn_estimator
from glob import glob
import os
import warnings

warnings.filterwarnings("ignore")


def closest(lst, k):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]


def del_activation_files():
    for f in glob("/tmp/*.txt"):
        os.remove(f)


def r2_score(y_test, predictions, y_hat):
    return 1 - np.mean((predictions - y_test) ** 2) / np.mean((y_test - y_hat) ** 2)


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


def data_plateau(nb_cols, nb_row, sigma2):
    dmin = -1
    dmax = 1
    r = dmax - dmin
    x = np.random.uniform(low=dmin, high=dmax, size=(nb_row, nb_cols))
    v_max = dmin + r / (2 ** (1 / nb_cols))
    y_true = [1 if x_val == nb_cols else 0 for x_val in (x <= v_max).sum(1)]
    y = y_true + np.random.normal(0, math.sqrt(sigma2), nb_row)

    return x, y


def calc_chi2_approx(p):
    chi2_approx = chi2.ppf(1 - (1 / 2 ** (1 / (np.ceil(n / p) - 1))), p - 1) / p - 1
    return chi2_approx


def calc_normal_approx(p, n, noise):
    factor = math.sqrt(2) * noise
    normal_approx = norm.ppf(1 - (1 / 2) ** (1 / (np.ceil(n / p) - 1)))
    normal_approx = noise + factor / math.sqrt(p) * normal_approx
    return normal_approx


def calc_logit_approx(p, n, noise):
    factor = math.sqrt(2) * noise
    logit = math.log(2 ** (1 / (np.ceil(n / p) - 1)) - 1)
    logit *= math.sqrt(math.pi / 8)
    logit *= factor / math.sqrt(p)
    return noise + logit


def unbiased_gessaman(p, n, estimator, var):
    f = norm.ppf(1 - (1 / 2) ** (1 / (np.ceil(n / p) - 1)))
    # deno = 1 + math.sqrt(2/p) * f
    # return m / deno
    return estimator - np.sqrt(var) * f


def unbiased_gessaman_old(p, n, m, var):
    f = norm.ppf(1 - (1 / 2) ** (1 / (np.ceil(n / p) - 1)))
    deno = 1 + math.sqrt(2/p) * f
    return m / deno
    # return m - var ** (1/2) * f


def find_gap(s):
    # gap = None
    # s -= np.average(s)
    # step = np.hstack((np.ones(len(s)), -1 * np.ones(len(s))))
    # s_step = np.convolve(s, step, mode='valid')
    # gap = np.argmax(s_step)  # TODO Find the first gap not the max !!!!

    s_diff = np.diff(s) / s[1:]
    # sub_diff = np.extract(s_diff > 0, s_diff)
    # gap_tests = list(s_diff > np.mean(sub_diff))
    temp = np.cumsum(s_diff) / np.array(range(1, len(s_diff) + 1))
    gap_tests = np.diff(temp) > 0
    if True in gap_tests:
        gaps = np.where(gap_tests)[0]
        for gap in gaps:
            if s_diff[gap + 2] == 0:
                return gap
        return -1
    else:
        return -1
    # gap = np.argmax(s_diff)
    # s_cum_mean = np.cumsum(s_diff) / np.array(range(1, len(s_diff)+1))
    # gaps = np.diff(s_cum_mean) < 0  # [g > m for g, m in zip(s_diff, s_cum_mean)]
    # gapidentifier un saut dans les donnéess_cumsum = list(np.cumsum(gaps))
    # gaps_cumsum = moving_average(np.round(s_diff, 2) >= 0, w)
    # plateau = s_diff == 0
    # gap_list = [False] + (s_diff > s_diff[0])[:-1]
    # gaps_cumsum = moving_average(plateau, w)
    # temp = [a*b for a, b in zip(gaps_cumsum, gap_list)]
    # if w in temp:
    #     # gap = list(gaps_cumsum).index(w)
    #     gap = temp.index(w)


def find_best_estimator(s, w):
    gap = find_gap(s, w)
    if gap is None:
        print('Last')
        return -1
    else:
        print(gap)
        return gap  # No -1 because the gap is calculated on the diff, hence we drop 1 dimension.


def extract_partition(rs, r_id):
    sub_rs = np.extract([len(r) == len(rs[r_id]) for r in rs], rs)
    sub_rs = np.extract([r.condition.features_indexes == rs[r_id].condition.features_indexes for r in sub_rs], sub_rs)
    return sub_rs


def do_qqplot(data, data_type, d):
    fig = sm.qqplot(data, line='45')
    fig.savefig(
        f"/home/vmargot/Documents/Jussieu/new/{data_type}_{n}_d={d}_qqplot",
        format="svg",
        dpi=300,
    )
    plt.close(fig)


def do_box_plots(sigma2_ratio_df, title, data_type, d, k_list):
    alpha = d / (2 * (d + 2))
    cov_min = k_list.index(closest(k_list, n ** (1-alpha)))

    fig = plt.figure(figsize=(25, 17))
    sns.boxplot(x="% of points", y="Sigma2 ratio", hue="Type", data=sigma2_ratio_df)
    plt.xlabel("% of points in each cell", fontsize=20)
    plt.ylabel("Sigma2 ratio", fontsize=20)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=20)
    plt.legend(loc="upper left", fontsize=15)
    plt.axhline(y=0, color="r", linestyle=":")
    plt.axvline(x=cov_min, color="r", linestyle=":")
    fig.savefig(
        f"/home/vmargot/Documents/Jussieu/new/{data_type}_{n}_d={d}_sigma2ratio",
        format="svg",
        dpi=300,
    )
    plt.close(fig)


def do_approx_graph(sigma2_ratio_df, k_list, title, data_type, n, d, do_approx):
    alpha = d / (2 * (d + 2))
    cov_min = k_list.index(closest(k_list, n ** (1-alpha)))

    temp = sigma2_ratio_df.loc[sigma2_ratio_df["Type"] == "Gessaman"][
        ["Sigma2 ratio", "% of points"]
    ].astype(float)
    m = temp.groupby("% of points")["Sigma2 ratio"].median()

    temp = sigma2_ratio_df.loc[sigma2_ratio_df["Type"] == "NN on sample"][
        ["Sigma2 ratio", "% of points"]
    ].astype(float)
    m2 = temp.groupby("% of points")["Sigma2 ratio"].median()
    q1 = temp.groupby("% of points")["Sigma2 ratio"].quantile(0.25)
    q3 = temp.groupby("% of points")["Sigma2 ratio"].quantile(0.75)

    alpha = 1 / 2
    th_min = k_list.index(closest(k_list, n ** (1-alpha)))

    sigma2_ratio_df = sigma2_ratio_df.loc[sigma2_ratio_df["Type"] == "Gessaman"]
    fig = plt.figure(figsize=(25, 17))
    sns.boxplot(x="% of points", y="Sigma2 ratio", hue="Type", data=sigma2_ratio_df)
    plt.axhline(y=0, color="k", linestyle=":")
    plt.axvline(x=cov_min, color="r", linestyle=":", label="a = d/(2(d+2))")
    plt.axvline(x=th_min, color="g", linestyle=":", label="a = 1/2")
    plt.plot(m.values, "--", linewidth=1, color='black', label="Gessaman median")
    plt.plot(m2.values, "--", linewidth=1, color="orange", label="1-NN median")
    plt.plot(q1.values, ":", linewidth=1, color="orange")
    plt.plot(q3.values, ":", linewidth=1, color="orange")
    if do_approx:
        # sigma2_estimates_m = (m + 1) * noise
        # sigma2_gessaman = (sigma2_ratio_df['Sigma2 ratio'] + 1) * noise

        vals_chi2 = [calc_chi2_approx(p) for p in k_list]
        vals_normal = [calc_normal_approx(p, n, noise) / noise - 1 for p in k_list]
        vals_logit = [calc_logit_approx(p, n, noise) / noise - 1 for p in k_list]
        # sigma2_approx = [unbiased_gessaman(p, n, m, np.var(sigma2_gessaman)) / noise - 1
        #                  for p, m in zip(k_list, sigma2_estimates_m.values)]

        plt.plot(vals_chi2, linestyle=(0, (5, 2, 1, 2)), linewidth=1, color="red", label="Chi2")
        plt.plot(vals_logit, linestyle=(0, (5, 2, 1, 2)), linewidth=1, color="green", label="Logit")
        plt.plot(vals_normal, linestyle=(0, (5, 2, 1, 2)), linewidth=1, color="blue", label="Normal")

        # plt.plot(sigma2_approx, linewidth=1.5, color="c", label="Gessaman unbiased (med)")

    plt.xlabel("% of points in each cell", fontsize=20)
    plt.ylabel("Sigma2 ratio", fontsize=20)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=20)
    plt.legend(loc="lower right", fontsize=15)
    fig.savefig(
        f"/home/vmargot/Documents/Jussieu/new/{data_type}_{n}_d={d}_sigma2ratio_approx",
        format="svg",
        dpi=300,
    )
    plt.close(fig)


def do_graph(n, noise, step, nb_simu, data_type, d=1, do_approx=False):
    # min_points = int(n * step)
    min_points = int(n**(1/2) + 1)
    k_list = list(range(min_points, int(n / 2), int(n * step)))

    types_list = ["Gessaman", "Gessaman unbiased", "NN on rule", "NN on sample"]
    resume_df_list = []
    g_unbiased_estim_list = []

    title = f"Points sensitivity for {data_type} dataset with sigma2={noise}, n={n} and for {nb_simu} simulations"
    for i in tqdm.tqdm(range(nb_simu)):
        resume_df = pd.DataFrame(index=list(range(len(k_list) * len(types_list))),
                                 columns=["% of points", "Sigma2 ratio", "Type"])
        types = types_list * len(k_list)
        resume_df['Type'] = np.sort(types)
        pct_points = list(np.round(np.array(k_list) / n * 100, 2))

        resume_df["% of points"] = pct_points * len(types_list)

        nn_estimate_on_rule_list = []
        gessaman_estimator_list = []
        gessaman_estimator_var = []
        pts_partition = []

        if data_type == "sigmoid":
            x, y = data_sigmoid(n, noise)
        elif data_type == "plateau":
            x, y = data_plateau(d, n, noise)
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

        for k in k_list:
            g = Gessaman(k=k, nb_jobs=4, verbose=False)
            g.fit(x, y, False)
            suitable_rs = g.significant_rs + g.insignificant_rs
            noise_estimators = [rule.std ** 2 for rule in suitable_rs]
            gessaman_estimate = min(noise_estimators)
            gessaman_estimator_list.append(gessaman_estimate)

            partition = extract_partition(suitable_rs, noise_estimators.index(gessaman_estimate))
            gessaman_estimator_var.append(np.var([rule.std ** 2 for rule in partition]))

            pts_partition.append(len(partition))
            nn_estimate_on_rule = min([rule._nn_estimate for rule in g.ruleset])
            nn_estimate_on_rule_list.append(nn_estimate_on_rule)

        best_id = find_gap(gessaman_estimator_list)
        print('Index of the estimator: ', str(best_id))
        g_unbiased_estim = unbiased_gessaman(p=k_list[best_id], n=n, estimator=gessaman_estimator_list[best_id],
                                             var=np.var(noise_estimators))
                                             # var=gessaman_estimator_var[best_id])
        g_unbiased_estim_list.append(g_unbiased_estim)

        neigh_estimator = nn_estimator.calc_1nn_noise_estimator(x, y)

        sigma2_ratio_list = list(np.array(gessaman_estimator_list) / noise - 1) +\
                            [g_unbiased_estim / noise - 1] * len(k_list)\
                            + list(np.array(nn_estimate_on_rule_list) / noise - 1) +\
                            [neigh_estimator / noise - 1] * len(k_list)
        resume_df["Sigma2 ratio"] = sigma2_ratio_list
        resume_df_list.append(resume_df)

    full_df = pd.concat(resume_df_list)
    do_box_plots(full_df, title, data_type, d=d, k_list=k_list)
    do_approx_graph(full_df, k_list, title, data_type, n, d, do_approx)

    del_activation_files()


if __name__ == "__main__":
    nb_simu = 10
    step = 0.02
    n_list = [1125, 2500, 5000]  # , 10000]  # , 20000]
    # n_list = [2500]
    exp_list = ['sigmoid', 'plateau', 'rulefit', 'flag', 'diag', 'circle']
    # exp_list = ['sigmoid']
    for exp in exp_list:
        # Choose among ['sigmoid', 'plateau', 'rulefit', 'flag', 'diag', 'circle']
        # The noise has been set to have a signal / noise ratio > 2
        if exp == 'sigmoid':
            noise = 0.05
            for n in n_list:
                do_graph(n, noise, step, nb_simu, "sigmoid")

        if exp == 'plateau':
            d_list = [2, 3, 5]  # , 8, 10]
            # d_list = [3]
            noise = 0.125
            for d in d_list:
                for n in n_list:
                    do_graph(n, noise, step, nb_simu, exp, d=d, do_approx=True)

        if exp == 'rulefit':
            d = 5
            noise = 0.3
            for n in n_list:
                do_graph(n, noise, step, nb_simu, exp, d)

        if exp == 'flag':
            d = 3
            noise = 1.0
            for n in n_list:
                do_graph(n, noise, step, nb_simu, exp, d)

        if exp == 'diag':
            d = 3
            noise = 1.2
            for n in n_list:
                do_graph(n, noise, step, nb_simu, exp, d)

        if exp == 'circle':
            d = 3
            noise = 1.5
            for n in n_list:
                do_graph(n, noise, step, nb_simu, "circle", d)
