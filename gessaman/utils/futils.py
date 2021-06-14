from typing import Union, List, Tuple
from functools import reduce
from itertools import permutations
import operator
import numpy as np
from scipy.stats.mstats import mquantiles
from joblib import Parallel, delayed

from ruleskit import RuleSet
from ruleskit import RegressionRule
from ruleskit import Activation
from ruleskit import HyperrectangleCondition


def eval_activation(rule, xs):
    return rule.evaluate(xs).raw


def inter_variance(rs: Union[RuleSet, List[RegressionRule]]):
    return np.var([r.prediction for r in rs])


def intra_variance(rs: Union[RuleSet, List[RegressionRule]]):
    return np.mean([r.std ** 2 for r in rs])


def conditional_mean(activation: Union[np.ndarray, None], y: np.ndarray) -> float:
    """Mean of all activated values

    If activation is None, we assume the given y have already been extracted from the activation vector,
    which saves time.
    """
    if activation is None:
        return float(np.nanmean(y))
    if sum(activation) > 0:
        if isinstance(activation, np.ndarray):
            y_conditional = np.extract(activation, y)
        else:
            raise TypeError("'activation' in conditional_mean must be None or a np.ndarray")
        if len(y_conditional) > 0:
            return float(np.nanmean(y_conditional))
    else:
        return 0.0


def get_permutation_list(seq: List, k: int) -> List[Tuple]:
    """
    Liste des arrangements des objets de la liste seq pris k à k
    Parameters
    ----------
    seq: List type
    k: int type

    Returns
    -------
    p: List[Tuple]

    Example
    -------
    >>> l1 = [1, 2, 3]
    >>> get_permutation_list(l1, 2)
    [(1, 2), (1, 3), (2, 1), (2, 3), (3, 2)]
    """
    p = []
    i, imax = 0, 2 ** len(seq) - 1
    while i <= imax:
        s = []
        j, jmax = 0, len(seq) - 1
        while j <= jmax:
            if (i >> j) & 1 == 1:
                s.append(seq[j])
            j += 1
        if len(s) == k:
            v = permutations(s)
            p.extend(v)
        i += 1
    return p


def get_pair(rls: List[List], is_identical: bool = False):
    """Get all possible pairs of elements between two lists. Will ignore Nones and same items.

    Parameters
    ---------
    rls: list
        Must contain two sublists
    is_identical: bool
        Pass True if the two elements of rls are the same lists, which will save a bit of computation time

    Example
    -------
    >>> l1 = [[1, 2], [1, 3, 4, 5, None]]
    >>> list(get_pair(l1))
    [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    """
    if is_identical:
        for i, r1 in enumerate(rls[0]):
            for r2 in rls[1][i:]:
                if not (r1 is None or r2 is None):
                    yield r1, r2
    else:
        for r1 in rls[0]:
            for r2 in rls[1]:
                if not (r1 is None or r2 is None):
                    yield r1, r2


def get_conditional_bins(
    x: np.ndarray,
    prob: List[float],
    col_id: int,
    row_id: int,
    bmaxs: Union[List, np.ndarray],
    bmins: Union[List, np.ndarray],
):
    """
    Parameters
    ----------
    x
    prob
    col_id
    row_id
    bmaxs
    bmins

    Returns
    -------

    """
    if col_id < x.shape[1]:
        bins = np.array(mquantiles(x[:, col_id], prob=prob, axis=0))
        for i in range(1, len(bins)):
            if bins[i] == max(x[:, col_id]):
                bmaxs[row_id, col_id] = np.Inf
            else:
                bmaxs[row_id, col_id] = bins[i]

            if bins[i - 1] == min(x[:, col_id]):
                bmins[row_id, col_id] = -np.Inf
            else:
                bmins[row_id, col_id] = bins[i - 1]
            if i == len(bins) - 1 or bins[i - 1] == bins[i]:
                new_x = x[(bins[i - 1] <= x[:, col_id]) & (x[:, col_id] <= bins[i]), :]
            else:
                new_x = x[(bins[i - 1] <= x[:, col_id]) & (x[:, col_id] < bins[i]), :]
            bmaxs, bmins, row_id = get_conditional_bins(
                new_x, prob, col_id + 1, row_id, bmaxs, bmins
            )
            if row_id < bmins.shape[0]:
                bmaxs[row_id, :col_id] = bmaxs[row_id - 1, :col_id]
                bmins[row_id, :col_id] = bmins[row_id - 1, :col_id]
    else:
        row_id += 1
    return bmaxs, bmins, row_id


def partition_space(x: np.ndarray, nb_cells: int, nb_dims: int):
    """
    Parameters
    ----------
    x
    nb_cells
    nb_dims

    Returns
    -------

    """
    cut_by_dim = int(pow(nb_cells, 1 / nb_dims))
    prob = [0 + i / cut_by_dim for i in range(0, cut_by_dim + 1)]
    nb_cells = cut_by_dim ** nb_dims
    bmaxs = np.zeros((nb_cells, nb_dims))
    bmins = np.zeros((nb_cells, nb_dims))
    bmaxs, bmins, _ = get_conditional_bins(x, prob, 0, 0, bmaxs, bmins)

    return bmaxs, bmins, nb_cells


def get_partition_rs(x, y, nb_dims, nb_cells, features_index):
    """
    Parameters
    ----------
    x
    y
    nb_dims
    nb_cells
    features_index

    Returns
    -------

    """
    sub_x = x[:, features_index]
    rs = RuleSet([])
    bmaxs, bmins, nb_cells = partition_space(sub_x, nb_cells, nb_dims)
    for i in range(int(nb_cells)):
        condition = HyperrectangleCondition(
            features_indexes=features_index,
            bmins=list(bmins[i]),
            bmaxs=list(bmaxs[i]),
            sort=False,
        )
        rule = RegressionRule(condition)
        rule.fit(x, y)
        rs += rule

    return rs, (features_index, inter_variance(rs))


def significant_test(rule, ymean, sigma2, beta):
    """
    Parameters
    ----------
    rule : {RiceRule type}
        A rule.

    ymean : {float type}
            The mean of y.

    sigma2 : {float type}
            The noise estimator.

    beta : {float type}
            The beta factor.

    Return
    ------
    The bound for the conditional expectation to be significant
    """
    left_term = beta * abs(rule.prediction - ymean)
    right_term = np.sqrt(max(0, rule.std ** 2 - sigma2))
    return left_term > right_term


def insignificant_test(rule, sigma2, epsilon):
    """
    Parameters
    ----------
    rule : {RiceRule type}
        A rule.

    sigma2 : {float type}
            The noise estimator.

    epsilon : {float type}
            The epsilon factor.

    Return
    ------
    The bound for the conditional expectation to be insignificant
    """
    return epsilon >= np.sqrt(max(0, rule.std ** 2 - sigma2))


def union_test(rule: RegressionRule, act: Activation, gamma=1.0):
    """
    Test to know if a rule (self) and an activation vector have
    at more gamma percent of points in common
    """
    # noinspection PyProtectedMember
    rule_activation = rule._activation
    intersect_vector = rule_activation & act

    pts_inter = intersect_vector.nones
    pts_act = act.nones
    pts_rule = rule_activation.nones

    ans = (pts_inter < gamma * pts_rule) and (pts_inter < gamma * pts_act)

    return ans


def select(rs: RuleSet, gamma: float, selected_rs: RuleSet = None) -> (int, RuleSet):
    """
    Parameters
    ----------
    rs
    gamma
    selected_rs

    Returns
    -------

    """
    i = 0
    rg_add = 0
    if selected_rs is None:
        selected_rs = rs[:1]
        rs = rs[1:]
        rg_add += 1

    nb_rules = len(rs)
    # old_criterion = calc_ruleset_crit(selected_rs, y_train, x_train, calcmethod)
    # crit_evo.append(old_criterion)
    while selected_rs.calc_coverage_rate() < 1 and i < nb_rules:
        new_rules = rs[i]
        # noinspection PyProtectedMember
        utests = [
            union_test(new_rules, rule._activation, gamma) for rule in selected_rs
        ]
        if all(utests) and union_test(new_rules, selected_rs.get_activation(), gamma):
            selected_rs += new_rules
            # old_criterion = new_criterion
            rg_add += 1
        # crit_evo.append(old_criterion)
        i += 1
    # self.set_params(critlist=crit_evo)
    return rg_add, selected_rs


def predict(rs: RuleSet, xs: np.ndarray, y_train: np.ndarray, nb_jobs: int = 2) -> (np.ndarray, np.ndarray):
    max_func = np.vectorize(max)
    significant_rules = list(filter(lambda rule: rule.significant, rs))
    insignificant_rules = list(filter(lambda rule: rule.significant is False, rs))

    if len(significant_rules) > 0:
        # noinspection PyProtectedMember
        significant_union = reduce(operator.add, [rule._activation for rule in significant_rules]).raw

        significant_act_matrix = [rule.activation for rule in significant_rules]
        significant_act_matrix = np.array(significant_act_matrix)
        significant_pred_matrix = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
            delayed(eval_activation)(rule, xs)for rule in significant_rules)
        significant_pred_matrix = np.array(significant_pred_matrix).T

        no_activation_matrix = np.logical_not(significant_pred_matrix)

        nb_rules_active = significant_pred_matrix.sum(axis=1)
        nb_rules_active[nb_rules_active == 0] = -1  # If no rule is activated

        # Activation of the intersection of all NOT activated rules at each row
        no_activation_vector = np.dot(no_activation_matrix, significant_act_matrix)
        no_activation_vector = np.array(no_activation_vector, dtype="int")

        # Activation of the intersection of all activated rules at each row
        dot_activation = np.dot(significant_pred_matrix, significant_act_matrix)
        dot_activation = np.array([np.equal(act, nb_rules) for act, nb_rules in zip(dot_activation, nb_rules_active)],
                                  dtype="int")

        # Calculation of the binary vector for cells of the partition et each row
        significant_cells = (dot_activation - no_activation_vector) > 0
        no_prediction_points = (significant_cells.sum(axis=1) == 0) & (significant_pred_matrix.sum(axis=1) != 0)

    else:
        significant_cells = np.zeros(shape=(xs.shape[0], len(y_train)), dtype="bool")
        significant_union = np.zeros(len(y_train))
        no_prediction_points = np.zeros(xs.shape[0])

    if len(insignificant_rules) > 0:
        # Activation of all rules in the learning set
        insignificant_act_matrix = [rule.activation for rule in insignificant_rules]
        insignificant_act_matrix = np.array(insignificant_act_matrix)
        insignificant_act_matrix -= significant_union
        insignificant_act_matrix = max_func(insignificant_act_matrix, 0)

        insignificant_pred_matrix = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
            delayed(eval_activation)(rule, xs) for rule in insignificant_rules)
        insignificant_pred_matrix = np.array(insignificant_pred_matrix).T

        no_activation_matrix = np.logical_not(insignificant_pred_matrix)

        nb_rules_active = insignificant_pred_matrix.sum(axis=1)
        nb_rules_active[nb_rules_active == 0] = -1  # If no rule is activated

        # Activation of the intersection of all NOT activated rules at each row
        no_activation_vector = np.dot(no_activation_matrix, insignificant_act_matrix)
        no_activation_vector = np.array(no_activation_vector, dtype="int")

        # Activation of the intersection of all activated rules at each row
        dot_activation = np.dot(insignificant_pred_matrix, insignificant_act_matrix)
        dot_activation = np.array([np.equal(act, nb_rules) for act, nb_rules in zip(dot_activation, nb_rules_active)],
                                  dtype="int",)

        # Calculation of the binary vector for cells of the partition et each row
        insignificant_cells = (dot_activation - no_activation_vector) > 0
    else:
        insignificant_cells = np.zeros(shape=(xs.shape[0], len(y_train)), dtype="bool")

    # Calculation of the No-rule prediction.
    no_rule_cell = np.ones(len(y_train)) - significant_union
    no_rule_prediction = conditional_mean(no_rule_cell, y_train)

    # Calculation of the conditional expectation in each cell
    cells = insignificant_cells ^ significant_cells
    prediction_vector = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
        delayed(conditional_mean)(act, y_train) for act in cells)
    prediction_vector = np.array(prediction_vector)
    prediction_vector[no_prediction_points] = np.nan
    prediction_vector[prediction_vector == 0] = no_rule_prediction

    return np.array(prediction_vector), no_prediction_points
