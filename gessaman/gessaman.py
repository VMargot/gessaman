from typing import List, Union
import numpy as np
from scipy.stats.mstats import mquantiles
from ruleskit import HyperrectangleCondition
from ruleskit import Rule
from ruleskit import RuleSet


def update_index(index):
    updated_index = [index[i + 1] + int(index[i] / len(index)) for i in range(len(index) - 1)]
    updated_index.append(index[-1])
    return updated_index


def get_conditional_bins(x: np.ndarray, prob: List[float], col_id: int, row_id: int,
                         bmaxs: Union[List, np.ndarray], bmins: Union[List, np.ndarray]):
    if col_id < x.shape[1]:
        bins = np.array(mquantiles(x[:, col_id], prob=prob, axis=0))
        for i in range(1, len(bins)):
            if bins[i] == max(x[:, col_id]):
                bmaxs[row_id, col_id] = np.Inf
            else:
                bmaxs[row_id, col_id] = bins[i]

            if bins[i-1] == min(x[:, col_id]):
                bmins[row_id, col_id] = -np.Inf
            else:
                bmins[row_id, col_id] = bins[i - 1]
            if i == len(bins)-1 or bins[i-1] == bins[i]:
                new_x = x[(bins[i-1] <= x[:, col_id]) & (x[:, col_id] <= bins[i]), :]
            else:
                new_x = x[(bins[i-1] <= x[:, col_id]) & (x[:, col_id] < bins[i]), :]
            bmaxs, bmins, row_id = get_conditional_bins(new_x, prob, col_id + 1, row_id, bmaxs, bmins)
            if row_id < bmins.shape[0]:
                bmaxs[row_id, :col_id] = bmaxs[row_id - 1, :col_id]
                bmins[row_id, :col_id] = bmins[row_id - 1, :col_id]
    else:
        row_id += 1
    return bmaxs, bmins, row_id


class Gessaman:
    """
    """
    def __init__(self, k: Union[int, None] = None):
        self._k = k
        self._n = None
        self._d = None
        self._nb_cells = None
        self._bmaxs = None
        self._bmins = None
        self._features_index = None
        self._ruleset = None
        pass

    @property
    def k(self) -> int:
        return self._k

    @property
    def n(self) -> int:
        return self._n

    @property
    def d(self) -> int:
        return self._d

    @property
    def bmaxs(self) -> np.ndarray:
        return self._bmaxs

    @property
    def bmins(self) -> np.ndarray:
        return self._bmins

    @property
    def nb_cells(self) -> int:
        return self._nb_cells

    @property
    def features_index(self) -> List:
        return self._features_index

    @property
    def ruleset(self) -> List:
        return self._ruleset

    @k.setter
    def k(self, value: int):
        self._k = value

    @n.setter
    def n(self, value: int):
        self._n = value

    @d.setter
    def d(self, value: int):
        self._d = value

    @nb_cells.setter
    def nb_cells(self, value: int):
        self._nb_cells = value

    @bmaxs.setter
    def bmaxs(self, value: np.ndarray):
        self._bmaxs = value

    @bmins.setter
    def bmins(self, value: np.ndarray):
        self._bmins = value

    @features_index.setter
    def features_index(self, value: List):
        self._features_index = value

    @ruleset.setter
    def ruleset(self, value: List):
        self._ruleset = value

    @staticmethod
    def get_dim_max(nb_cells: int, d: int) -> int:
        nb_dims = d
        while pow(nb_cells, 1 / nb_dims) < 2:
            nb_dims -= 1
        return nb_dims

    @staticmethod
    def select_dims(x: np.ndarray, nb_dims: int) -> List:
        corr_mat = np.corrcoef(x.T)
        dims_list = [0]
        for d in dims_list:
            corr_vector = abs(corr_mat[dims_list].sum(axis=0))
            dim = np.argmin(abs(corr_vector))
            while dim in dims_list:
                corr_vector[dim] = 1
                dim = np.argmin(abs(corr_vector))
            dims_list.append(dim)
            if len(dims_list) == nb_dims:
                break
        dims_list.sort()
        return dims_list

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

    def partition_space(self, x: np.ndarray, k: int):
        nb_dims = self.get_dim_max(self.nb_cells, self.d)
        if nb_dims < self.d:
            features_index = self.select_dims(x, nb_dims)
            x = x[:, features_index]
        else:
            features_index = list(range(self.d))
        cut_by_dim = int(pow(self.nb_cells, 1 / nb_dims))
        prob = [0 + i / cut_by_dim for i in range(0, cut_by_dim + 1)]
        self.nb_cells = cut_by_dim**nb_dims
        bmaxs = np.zeros((cut_by_dim**nb_dims, nb_dims))
        bmins = np.zeros((cut_by_dim ** nb_dims, nb_dims))
        bmaxs, bmins, _ = get_conditional_bins(x, prob, 0, 0, bmaxs, bmins)

        return bmaxs, bmins, features_index

    def get_rules(self, x, y):
        rules_list = []
        for i in range(int(self.nb_cells)):
            condition = HyperrectangleCondition(features_indexes=self.features_index,
                                                bmins=list(self.bins[i]),
                                                bmaxs=list(self.bmaxs[i]))
            rule = Rule(condition)
            rule.fit(x, y)
            rules_list.append(rule)

        return RuleSet(rules_list)

    def fit(self, x: np.ndarray, y: np.ndarray):
        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y must have the same number of observations.')
        self.n, self.d = x.shape
        if self.k is None:
            self.k = int(pow(self.n, 1/3))
        elif self.k > self.n:
            raise ValueError('The number of bucket must smaller than the number of observations.')
        self.nb_cells = np.ceil(self.n / self.k)

        self.bmaxs, self.bins, self.features_index = self.partition_space(x, self.k)
        self.ruleset = self.get_rules(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.ruleset is None:
            raise ValueError('The model is not fitted.')
        if x.shape[1] != self.d:
            raise ValueError('The dimension of x is different than the training set.')
        pred_list = [rule.predict(x) for rule in self.ruleset]
        nb_activated_rules = np.sum([p.astype('bool') for p in pred_list], axis=0)
        return np.sum(pred_list, axis=0) / nb_activated_rules
