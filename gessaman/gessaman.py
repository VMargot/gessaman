from typing import List, Union, Tuple
from functools import reduce
import operator
from os import cpu_count
import numpy as np
from joblib import Parallel, delayed
from ruleskit import RuleSet
from .utils import futils as f


class Gessaman:
    """ """

    def __init__(self, gamma: float = 0.8, nb_jobs: int = -1):
        self._n = None
        self._d = None
        self._beta = None
        self._epsilon = None
        self._sigma2 = None
        self._gamma = gamma
        if nb_jobs == -1 or nb_jobs > cpu_count():
            self._nbjobs = cpu_count()
        else:
            self._nbjobs = nb_jobs
        self._nb_cells = None
        self._nb_dims = None
        self._ruleset = RuleSet([])
        self._selected_rs = RuleSet([])
        self._partition_bvar = []

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
    def sigma2(self) -> float:
        return self._sigma2

    @property
    def nb_cells(self) -> int:
        return self._nb_cells

    @property
    def ruleset(self) -> Union[RuleSet, None]:
        return self._ruleset

    @property
    def selected_rs(self) -> Union[RuleSet, None]:
        return self._selected_rs

    @property
    def partition_bvar(self) -> List[Tuple]:
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

    @sigma2.setter
    def sigma2(self, value: float):
        self._sigma2 = value

    @nb_dims.setter
    def nb_dims(self, value: int):
        self._nb_dims = value

    @nb_cells.setter
    def nb_cells(self, value: int):
        self._nb_cells = value

    @ruleset.setter
    def ruleset(self, value: RuleSet):
        self._ruleset = value

    @selected_rs.setter
    def selected_rs(self, value: RuleSet):
        self._selected_rs = value

    @partition_bvar.setter
    def partition_bvar(self, value: List[Tuple]):
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

    def get_rules(self, x: np.ndarray, y: np.ndarray, nb_dims: int, rs: RuleSet) -> RuleSet:
        feats = list(range(self.d))
        nb_cells = self.nb_cells
        if nb_dims == 1:
            features_index_list = [[i] for i in feats]
        else:
            features_index_list = f.get_permutation_list(feats, nb_dims)

        rep = Parallel(n_jobs=self._nbjobs, backend="multiprocessing")(
            delayed(f.get_partition_rs)(x, y, nb_dims, nb_cells, features_index)
            for features_index in features_index_list
        )

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

    def fit(self, x: np.ndarray, y: np.ndarray):
        # --------------
        # DESIGNING PART
        # --------------
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of observations.")
        self.n, self.d = x.shape
        alpha = (3 * self.d + 1) / (6 * (self.d + 1))
        self.nb_cells = int(pow(self.n, alpha))
        self.nb_dims = min(self.d, int(np.log(self.n) / (2 * np.log(2)) - 1))

        print("----- Design rules ------")
        for lg in range(1, self.nb_dims + 1):
            self.ruleset = self.get_rules(x, y, lg, self.ruleset)

        # --------------
        # SELECTION PART
        # --------------
        print("----- Selection ------")
        self.beta = 1.0 / pow(self.d, 1.0 / 4 - alpha / 2.0)
        self.epsilon = self.beta * np.std(y)
        if self.sigma2 is None:
            self.sigma2 = min([rule.std ** 2 for rule in self.ruleset])

        self.selected_rs = self.select_rules(y.mean())

    def select_rules(self, ymean: float):
        """
        Returns a subset of a given ruleset.
        This subset minimizes the empirical contrast on the learning set
        """
        selected_rs = RuleSet([])
        beta = self.beta
        epsilon = self.epsilon
        sigma2 = self.sigma2
        gamma = self.gamma
        rs = self.ruleset
        print("Number of rules: %s" % str(len(rs)))

        # Selection of significant rules
        significant_rules = list(filter(lambda rule: f.significant_test(rule, ymean, sigma2, beta), rs))
        if len(significant_rules) > 0:
            [setattr(rule, "significant", True) for rule in significant_rules]
            print(f"Number of rules after significant test: {len(significant_rules)}")
            sorted_significant = sorted(significant_rules, key=lambda x: x.coverage, reverse=True)
            significant_rs = RuleSet(list(sorted_significant))

            rg_add, selected_rs = f.select(significant_rs, gamma)
            print("Number of selected significant rules: %s" % str(rg_add))

        else:
            print("No significant rules selected!")

        # Add insignificant rules to the current selection set of rules
        if selected_rs is None or selected_rs.calc_coverage_rate() < 1:
            insignificant_list = filter(lambda rule: f.insignificant_test(rule, sigma2, epsilon), rs)
            insignificant_list = list(filter(lambda rule: rule not in significant_rules, insignificant_list))
            if len(list(insignificant_list)) > 0:
                [setattr(rule, "significant", False) for rule in insignificant_list]
                print(f"Number rules after insignificant test: {len(insignificant_list)}")
                insignificant_list = sorted(insignificant_list, key=lambda x: x.std, reverse=False)
                insignificant_rs = RuleSet(list(insignificant_list))
                rg_add, selected_rs = f.select(insignificant_rs, gamma, selected_rs)
                print("Number insignificant rules added: %s" % str(rg_add))
            else:
                print("No insignificant rule added.")
        else:
            print("Covering is completed. No insignificant rule added.")

        # Add rule to have a covering
        coverage_rate = selected_rs.calc_coverage_rate()
        if coverage_rate < 1:
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
            print("Covering is completed.")

        return selected_rs

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
        selected_rs = self.selected_rs

        prediction_vector, no_predictions = f.predict(selected_rs, xs, y_train, nb_jobs=self._nbjobs)
        print(f"There are {round(sum(no_predictions) / xs.shape[0] * 100, 2)}% of observations without prediction.")
        return prediction_vector, no_predictions

    # def predict(self, x: np.ndarray) -> np.ndarray:
    #     if self.ruleset is None:
    #         raise ValueError('The model is not fitted.')
    #     if x.shape[1] != self.d:
    #         raise ValueError('The dimension of x is different than the training set.')
    #     pred_list = [rule.predict(x) for rule in self.ruleset]
    #     nb_activated_rules = np.sum([p.astype('bool') for p in pred_list], axis=0)
    #     return np.sum(pred_list, axis=0) / nb_activated_rules
