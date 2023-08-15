import numpy as np
from sklearn.metrics import pairwise_distances
import tslearn.metrics
from tslearn.barycenters import (
    euclidean_barycenter,
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    softdtw_barycenter,
)
from hyppo.independence import (
    MGC,
    Dcorr,
    HHG,
    Hsic,
)
from hyppo.time_series import MGCX, DcorrX

from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    Signed,
    parse_bivariate,
    parse_multivariate,
)

from itertools import combinations, permutations
from joblib import Parallel, delayed
from tqdm import tqdm

class PairwiseDistance(Undirected, Unsigned):

    name = "Pairwise distance"
    identifier = "pdist"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, metric="euclidean", **kwargs):
        self._metric = metric
        self.identifier += f"_{metric}"

    @parse_multivariate
    def multivariate(self, data):
        return pairwise_distances(data.to_numpy(squeeze=True), metric=self._metric)


""" TODO: include optional kernels in each method
"""


class HilbertSchmidtIndependenceCriterion(Undirected, Unsigned):
    """Hilbert-Schmidt Independence Criterion (HSIC)"""

    name = "Hilbert-Schmidt Independence Criterion"
    identifier = "hsic"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_multivariate
    def multivariate(self, data):

        def _get_hsic(pair, mat):
            return pair[0], pair[1], Hsic(bias=self._biased).statistic(mat[pair[0], :].reshape(-1, 1), mat[pair[1], :].reshape(-1, 1))
        
        pres = Parallel(n_jobs=-1)(delayed(_get_hsic)(pair, data.to_numpy()) for pair in tqdm(combinations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]
        res_mat += res_mat.T

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = Hsic(bias=self._biased).statistic(x, y)
        return stat


class HellerHellerGorfine(Directed, Unsigned):
    """Heller-Heller-Gorfine independence criterion"""

    name = "Heller-Heller-Gorfine Independence Criterion"
    identifier = "hhg"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "directed"]

    @parse_multivariate
    def multivariate(self, data):

        def _get_hhg(pair, mat):
            return pair[0], pair[1], HHG().statistic(mat[pair[0], :], mat[pair[1], :])
        
        pres = Parallel(n_jobs=-1)(delayed(_get_hhg)(pair, data.to_numpy()) for pair in tqdm(permutations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]

        return res_mat
    
    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = HHG().statistic(x, y)
        return stat


class DistanceCorrelation(Undirected, Unsigned):
    """Distance correlation"""

    name = "Distance correlation"
    identifier = "dcorr"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier += "_biased"

    @parse_multivariate
    def multivariate(self, data):

        def _get_dcorr(pair, mat):
            return pair[0], pair[1], Dcorr(bias=self._biased).statistic(mat[pair[0], :], mat[pair[1], :])
        
        pres = Parallel(n_jobs=-1)(delayed(_get_dcorr)(pair, data.to_numpy()) for pair in tqdm(combinations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]
        res_mat += res_mat.T

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """ """
        x, y = data.to_numpy()[[i, j]]
        stat = Dcorr(bias=self._biased).statistic(x, y)
        return stat


class MultiscaleGraphCorrelation(Undirected, Unsigned):
    """Multiscale graph correlation"""

    name = "Multiscale graph correlation"
    identifier = "mgc"
    labels = ["distance", "unsigned", "unordered", "nonlinear", "undirected"]

    @parse_multivariate
    def multivariate(self, data):

        def _get_mgc(pair, mat):
            return pair[0], pair[1], MGC().statistic(mat[pair[0], :], mat[pair[1], :])
        
        pres = Parallel(n_jobs=-1)(delayed(_get_mgc)(pair, data.to_numpy()) for pair in tqdm(combinations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]
        res_mat += res_mat.T

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = MGC().statistic(x, y)
        return stat


class CrossDistanceCorrelation(Directed, Unsigned):
    """Cross-distance correlation"""

    name = "Cross-distance correlation"
    identifier = "dcorrx"
    labels = ["distance", "unsigned", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{max_lag}"

    @parse_multivariate
    def multivariate(self, data):

        def _get_dcorrx(pair, mat):
            return pair[0], pair[1], DcorrX(max_lag=self._max_lag).statistic(mat[pair[0], :], mat[pair[1], :])[0]
        
        pres = Parallel(n_jobs=-1)(delayed(_get_dcorrx)(pair, data.to_numpy()) for pair in tqdm(permutations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        stat, _ = DcorrX(max_lag=self._max_lag).statistic(x, y)
        return stat


class CrossMultiscaleGraphCorrelation(Directed, Unsigned):
    """Cross-multiscale graph correlation"""

    name = "Cross-multiscale graph correlation"
    identifier = "mgcx"
    labels = ["unsigned", "distance", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier += f"_maxlag-{max_lag}"

    @parse_multivariate
    def multivariate(self, data):

        def _get_mgcx(pair, mat):
            return pair[0], pair[1], MGCX(max_lag=self._max_lag).statistic(mat[pair[0], :], mat[pair[1], :])[0]
        
        pres = Parallel(n_jobs=-1)(delayed(_get_mgcx)(pair, data.to_numpy()) for pair in tqdm(permutations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        stat, _, _ = MGCX(max_lag=self._max_lag).statistic(x, y)
        return stat


class TimeWarping(Undirected, Unsigned):

    labels = ["unsigned", "distance", "temporal", "undirected", "nonlinear"]

    def __init__(self, global_constraint=None):
        gcstr = global_constraint
        if gcstr is not None:
            gcstr = gcstr.replace("_", "-")
            self.identifier += f"_constraint-{gcstr}"
        self._global_constraint = global_constraint

    @property
    def simfn(self):
        try:
            return self._simfn
        except AttributeError:
            raise NotImplementedError(
                f"Add the similarity function for {self.identifier}"
            )

    @parse_multivariate
    def multivariate(self, data):
        def _get_dtw(pair, mat):
            return pair[0], pair[1], self._simfn(mat[pair[0], :], mat[pair[1], :], global_constraint=self._global_constraint)
        
        pres = Parallel(n_jobs=-1)(delayed(_get_dtw)(pair, data.to_numpy(squeeze=True)) for pair in tqdm(combinations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]
        res_mat += res_mat.T

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        return self._simfn(z[i], z[j], global_constraint=self._global_constraint)


class DynamicTimeWarping(TimeWarping):

    name = "Dynamic time warping"
    identifier = "dtw"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.dtw


class LongestCommonSubsequence(TimeWarping):

    name = "Longest common subsequence"
    identifier = "lcss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.lcss


class SoftDynamicTimeWarping(TimeWarping):

    name = "Dynamic time warping"
    identifier = "softdtw"

    @parse_multivariate
    def multivariate(self, data):
        def _get_softdtw(pair, mat):
            return pair[0], pair[1], tslearn.metrics.soft_dtw(mat[pair[0], :], mat[pair[1], :])
        
        pres = Parallel(n_jobs=-1)(delayed(_get_softdtw)(pair, data.to_numpy(squeeze=True)) for pair in tqdm(combinations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]
        res_mat += res_mat.T

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.soft_dtw(z[i], z[j])


class Barycenter(Directed, Signed):

    name = "Barycenter"
    identifier = "bary"
    labels = ["distance", "signed", "undirected", "temporal", "nonlinear"]

    def __init__(self, mode="euclidean", squared=False, statistic="mean"):
        if mode == "euclidean":
            self._fn = euclidean_barycenter
        elif mode == "dtw":
            self._fn = dtw_barycenter_averaging
        elif mode == "sgddtw":
            self._fn = dtw_barycenter_averaging_subgradient
        elif mode == "softdtw":
            self._fn = softdtw_barycenter
        else:
            raise NameError(f"Unknown Barycenter mode: {mode}")
        self._mode = mode

        self._squared = squared
        self._preproc = lambda x: x
        if squared:
            self._preproc = lambda x: x**2
            self.identifier += f"_sq"

        if statistic == "mean":
            self._statfn = lambda x: np.nanmean(self._preproc(x))
        elif statistic == "max":
            self._statfn = lambda x: np.nanmax(self._preproc(x))
        else:
            raise NameError(f"Unknown statistic: {statistic}")

        self.identifier += f"_{mode}_{statistic}"

    @parse_multivariate
    def multivariate(self, data):
        def _get_barycenter(pair, mat):
            return pair[0], pair[1], self._fn(mat[[pair[0], pair[1]]])
        
        try:
            pres = data.barycenter[self._mode]
        except (AttributeError, KeyError):
            pres = Parallel(n_jobs=-1)(delayed(_get_barycenter)(pair, data.to_numpy(squeeze=True)) for pair in tqdm(combinations(range(data.n_processes), 2)))
            try:
                data.barycenter[self._mode] = pres
            except AttributeError: # first run, nothing in data.barycenter
                data.barycenter = {self._mode: pres}

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = self._statfn(p[2])
        res_mat += res_mat.T

        return res_mat


    @parse_bivariate
    def bivariate(self, data, i=None, j=None):

        try:
            bc = data.barycenter[self._mode][(i, j)]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True)
            bc = self._fn(z[[i, j]])
            try:
                data.barycenter[self._mode][(i, j)] = bc
            except AttributeError:
                data.barycenter = {self._mode: {(i, j): bc}}
            except KeyError:
                data.barycenter[self._mode] = {(i, j): bc}
            data.barycenter[self._mode][(j, i)] = data.barycenter[self._mode][(i, j)]

        return self._statfn(bc)
