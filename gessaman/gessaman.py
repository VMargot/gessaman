import copy
from typing import List, Union
from functools import reduce
import operator
from os import cpu_count
import numpy as np
from joblib import Parallel, delayed
from ruleskit import RuleSet
from .utils import futils as f
from .utils.cell import BaseCell


def get_significant_rs(rs, ymean, sigma2, beta):
    significant_rules = list(
        filter(lambda rule: f.significant_test(rule, ymean, sigma2, beta), rs)
    )
    sorted_significant = sorted(
        significant_rules, key=lambda x: x.coverage, reverse=True
    )
    significant_rs = RuleSet(list(sorted_significant))
    return significant_rs


def get_insignificant_rs(rs, sigma2, epsilon, significant_rules=RuleSet([])):
    insignificant_list = filter(
        lambda rule: f.insignificant_test(rule, sigma2, epsilon), rs
    )
    insignificant_list = list(
        filter(lambda rule: rule not in significant_rules, insignificant_list)
    )
    insignificant_list = sorted(
        insignificant_list, key=lambda x: x.std, reverse=False
    )
    insignificant_rs = RuleSet(list(insignificant_list))
    return insignificant_rs


class Gessaman:
    """ """

    def __init__(
        self,
        alpha: Union[float, None] = None,
        k: Union[int, None] = None,
        gamma: float = 0.8,
        nb_jobs: int = -1,
        verbose: bool = True,
    ):
        BaseCell.instances = []
        self.verbose = verbose

        self._n = None
        self._d = None
        self._beta = None
        self._epsilon = None
        self._sigma2 = None

        self._alpha = alpha
        self._gamma = gamma
        self._k = k
        if nb_jobs == -1 or nb_jobs > cpu_count():
            self._nbjobs = cpu_count()
        else:
            self._nbjobs = nb_jobs

        self._nb_cells = None
        self._nb_dims = None
        self._partition_bvar = []

        self.ruleset = RuleSet([])
        self.selected_rs = RuleSet([])
        self.significant_rs = RuleSet([])
        self.insignificant_rs = RuleSet([])

    @property
    def n(self) -> int:
        return self._n

    @property
    def d(self) -> int:
        return self._d

    @property
    def nb_dims(self) -> int:
        return self._nb_dims

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def alpha(self) -> Union[float, None]:
        return self._alpha

    @property
    def k(self) -> Union[int, None]:
        return self._k

    @property
    def sigma2(self) -> float:
        return self._sigma2

    @property
    def nb_cells(self) -> int:
        return self._nb_cells

    @property
    def partition_bvar(self) -> List[List]:
        return self._partition_bvar

    @n.setter
    def n(self, value: int):
        self._n = value

    @d.setter
    def d(self, value: int):
        self._d = value

    @beta.setter
    def beta(self, value: float):
        self._beta = value

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    @gamma.setter
    def gamma(self, value: float):
        self._gamma = value

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

    @k.setter
    def k(self, value: int):
        self._k = value

    @sigma2.setter
    def sigma2(self, value: float):
        self._sigma2 = value

    @nb_dims.setter
    def nb_dims(self, value: int):
        self._nb_dims = value

    @nb_cells.setter
    def nb_cells(self, value: int):
        self._nb_cells = value

    @partition_bvar.setter
    def partition_bvar(self, value: List[List]):
        self._partition_bvar = value

    @staticmethod
    def select_dims(nb_dims: int) -> List:
        pass
        # corr_mat = np.corrcoef(x.T)
        # dims_list = [0]
        # for d in dims_list:
        #     corr_vector = abs(corr_mat[dims_list].sum(axis=0))
        #     dim = np.argmin(abs(corr_vector))
        #     while dim in dims_list:
        #         corr_vector[dim] = 1
        #         dim = np.argmin(abs(corr_vector))
        #     dims_list.append(dim)
        #     if len(dims_list) == nb_dims:
        #         break
        # dims_list.sort()
        # return dims_list

    # @staticmethod
    # def get_bins(x: np.ndarray, features_index: List[int], prob: List[float]):
    #     bins_dict = {}
    #     bins = np.array(mquantiles(x[:, features_index[0]], prob=prob, axis=0))
    #     for b in bins:
    #
    #
    #     for col in features_index:
    #         bins = np.array(mquantiles(x[:, col], prob=prob, axis=0))
    #
    #     bins = np.vstack((x[:, col].min(axis=0), bins, x[:, col].max(axis=0)))
    #     return bins

    def get_rules(
        self, x: np.ndarray, y: np.ndarray, nb_dims: int, rs: RuleSet
    ) -> RuleSet:
        feats = list(range(self.d))
        nb_cells = self.nb_cells
        # print('Number of cells of the partition:', nb_cells)
        if nb_dims == 1:
            features_index_list = [[i] for i in feats]
        else:
            features_index_list = f.get_permutation_list(feats, nb_dims)

        rep = Parallel(n_jobs=self._nbjobs, backend="multiprocessing")(
            delayed(f.get_partition_rs)(x, y, nb_dims, nb_cells, features_index)
            for features_index in features_index_list
        )  # tqdm.tqdm(features_index_list))

        self.partition_bvar += [r[1] for r in rep]
        rs += reduce(operator.add, [r[0] for r in rep])

        # for features_index in features_index_list:
        #     partition_rs = self.get_partition_rs(x, y, nb_dims, features_index)
        #
        #     if rs is None:
        #         rs = partition_rs
        #     else:
        #         rs += partition_rs

        return rs

    def fit(self, x: np.ndarray, y: np.ndarray, do_selection: True):
        # --------------
        # DESIGNING PART
        # --------------
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of observations.")
        self.n, self.d = x.shape
        if self.alpha is None:
            self.alpha = (3 * self.d + 1) / (6 * (self.d + 1))

        if self.k is None:
            self.nb_cells = int(self.n ** self.alpha)
        else:
            assert (
                self.k <= self.n
            ), "The number of points in cells is greater than the number of observation!"
            self.nb_cells = int(self.n / self.k)

        self.nb_dims = min(self.d, int(np.log(self.n) / (2 * np.log(2)) - 1))
        self.beta = 1.0 / pow(self.d, 1.0 / 4 - self.alpha / 2.0)
        self.epsilon = self.beta * np.std(y)
        # self.nb_dims = min(self.d, 2)

        if self.verbose:
            print("----- Design rules ------")
        for lg in range(1, self.nb_dims + 1):
            self.ruleset = self.get_rules(x, y, lg, self.ruleset)
        if self.verbose:
            print(f"Number of rules: {len(self.ruleset)}")
        if self.sigma2 is None:
            self.sigma2 = min([rule.std ** 2 for rule in self.ruleset])

        self.significant_rs = get_significant_rs(self.ruleset, y.mean(), self.sigma2, self.beta)
        self.insignificant_rs = get_insignificant_rs(self.ruleset, self.sigma2, self.epsilon, self.significant_rs)

        # --------------
        # SELECTION PART
        # --------------
        if do_selection:
            if self.verbose:
                print("----- Selection ------")
            (self.selected_rs,
             self.selected_significant_rs,
             self.selected_insignificant_rs)\
                = self.select_rules(self.significant_rs, self.insignificant_rs)

    def select_rules(self, significant_rs, insignificant_rs):
        """
        Returns a subset of a given ruleset.
        This subset minimizes the empirical contrast on the learning set
        """
        selected_rs = RuleSet([])
        gamma = self.gamma

        # Selection of significant rules
        if len(significant_rs) > 0:
            rg_add, selected_rs = f.select(significant_rs, gamma)
            selected_significant_rs = copy.copy(selected_rs)
            if self.verbose:
                print(f"Number of selected significant rules: {rg_add}")

        else:
            selected_significant_rs = RuleSet([])
            if self.verbose:
                print("No significant rules selected!")

        # Add insignificant rules to the current selection set of rules
        if selected_rs is None or selected_rs.coverage < 1:
            if len(list(insignificant_rs)) > 0:
                rg_add, selected_rs = f.select(insignificant_rs, gamma, selected_rs)
                insignificant_rules_added = list(filter(lambda r: r not in selected_significant_rs, selected_rs))
                selected_insignificant_rs = RuleSet(insignificant_rules_added)
                if self.verbose:
                    print(f"Number insignificant rules added: {rg_add}")
            else:
                selected_insignificant_rs = RuleSet([])
                if self.verbose:
                    print("No insignificant rule added.")
        else:
            selected_insignificant_rs = RuleSet([])
            if self.verbose:
                print("Covering is completed. No insignificant rule added.")

        # Add rule to have a covering
        coverage_rate = selected_rs.coverage
        if coverage_rate < 1:
            if self.verbose:
                print("Warning: Covering is not completed!", coverage_rate)
            # neg_rule, pos_rule = add_no_rule(selected_rs, x_train, y_train)
            # features_name = self.get_param('features_name')
            #
            # if neg_rule is not None:
            #     id_feature = neg_rule.conditions.get_param('features_index')
            #     rule_features = list(itemgetter(*id_feature)(features_name))
            #     neg_rule.conditions.set_params(features_name=rule_features)
            #     neg_rule.calc_stats(y=y_train, x=x_train, cov_min=0.0, cov_max=1.0)
            #     print('Add negative no-rule  %s.' % str(neg_rule))
            #     selected_rs.append(neg_rule)
            #
            # if pos_rule is not None:
            #     id_feature = pos_rule.conditions.get_param('features_index')
            #     rule_features = list(itemgetter(*id_feature)(features_name))
            #     pos_rule.conditions.set_params(features_name=rule_features)
            #     pos_rule.calc_stats(y=y_train, x=x_train, cov_min=0.0, cov_max=1.0)
            #     print('Add positive no-rule  %s.' % str(pos_rule))
            #     selected_rs.append(pos_rule)
        else:
            if self.verbose:
                print("Covering is completed.")

        return selected_rs, selected_significant_rs, selected_insignificant_rs

    def predict(self, xs, y_train: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        application of the selected ruleset on X.

        Parameters
        ----------
        xs : {array type or sparse matrix of shape = [n_samples, n_features]}
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a spares matrix is provided, it will be
            converted into a spares ``csr_matrix``.

        y_train : {array type or sparse matrix of shape = [n_samples]}

        Returns
        -------
        y : {array type of shape = [n_samples]}
            The predicted values.
        """
        selected_significant_rs = self.selected_significant_rs
        selected_insignificant_rs = self.selected_insignificant_rs

        prediction_vector, no_predictions = f.predict(
            selected_significant_rs,
            selected_insignificant_rs,
            xs,
            y_train,
            nb_jobs=self._nbjobs,
        )
        if self.verbose:
            print(
                f"There are {round(sum(no_predictions) / xs.shape[0] * 100, 2)}% of observations without prediction."
            )
        return prediction_vector, no_predictions

    # def predict(self, x: np.ndarray) -> np.ndarray:
    #     if self.ruleset is None:
    #         raise ValueError('The model is not fitted.')
    #     if x.shape[1] != self.d:
    #         raise ValueError('The dimension of x is different than the training set.')
    #     pred_list = [rule.predict(x) for rule in self.ruleset]
    #     nb_activated_rules = np.sum([p.astype('bool') for p in pred_list], axis=0)
    #     return np.sum(pred_list, axis=0) / nb_activated_rules
