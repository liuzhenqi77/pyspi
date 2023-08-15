import warnings
import numpy as np

from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import mne_connectivity as mnec

from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)

from itertools import combinations, permutations
from joblib import Parallel, delayed
from tqdm import tqdm

class Cointegration(Undirected, Unsigned):

    name = "Cointegration"
    identifier = "coint"
    labels = ["misc", "unsigned", "temporal", "undirected", "nonlinear"]

    def __init__(
        self,
        method="johansen",
        statistic="trace_stat",
        det_order=1,
        k_ar_diff=1,
        autolag="aic",
        maxlag=10,
        trend="c",
    ):
        self._method = method
        self._statistic = statistic
        if method == "johansen":
            self.identifier += (
                f"_{method}_{statistic}_order-{det_order}_ardiff-{k_ar_diff}"
            )
            self._det_order = det_order
            self._k_ar_diff = k_ar_diff
        else:
            self._autolag = autolag
            self._maxlag = maxlag
            self._trend = trend
            self.identifier += (
                f"_{method}_{statistic}_trend-{trend}_autolag-{autolag}_maxlag-{maxlag}"
            )

    @property
    def key(self):
        key = (self._method,)
        if self._method == "johansen":
            return key + (self._det_order, self._k_ar_diff)
        else:
            return key + (self._autolag, self._maxlag, self._trend)

    def _from_cache(self, data, i, j):
        idx = (i, j)
        try:
            ci = data.coint[self.key][idx]
        except (KeyError, AttributeError):
            z = data.to_numpy(squeeze=True)

            if self._method == "aeg":
                stats = stattools.coint(
                    z[i],
                    z[j],
                    autolag=self._autolag,
                    maxlag=self._maxlag,
                    trend=self._trend,
                )

                ci = {"tstat": stats[0]}
            else:
                stats = coint_johansen(
                    z[[i, j]].T, det_order=self._det_order, k_ar_diff=self._k_ar_diff
                )

                ci = {
                    "max_eig_stat": stats.max_eig_stat[0],
                    "trace_stat": stats.trace_stat[0],
                }

            try:
                data.coint[self.key][idx] = ci
            except AttributeError:
                data.coint = {self.key: {idx: ci}}
            except KeyError:
                data.coint[self.key] = {idx: ci}
            data.coint[self.key][(j, i)] = ci

        return ci

    # Return the negative t-statistic (proxy for how co-integrated they are)
    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        ci = self._from_cache(data, i, j)
        return ci[self._statistic]


class LinearModel(Directed, Unsigned):
    name = "Linear model regression"
    identifier = "lmfit"
    labels = ["misc", "unsigned", "unordered", "normal", "linear", "directed"]

    def __init__(self, model):
        self.identifier += f"_{model}"
        self._model = getattr(linear_model, model)

    @parse_multivariate
    def multivariate(self, data):
        def _get_lmfit(pair, mat):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = self._model().fit(mat[pair[0]], np.ravel(mat[pair[1]]))
            y_predict = mdl.predict(mat[pair[0]])
            return pair[0], pair[1], mean_squared_error(y_predict, np.ravel(mat[pair[1]]))
        
        pres = Parallel(n_jobs=-1)(delayed(_get_lmfit)(pair, data.to_numpy()) for pair in tqdm(permutations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]
        res_mat += res_mat.T

        return res_mat


    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = self._model().fit(z[i], np.ravel(z[j]))
        y_predict = mdl.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))


class GPModel(Directed, Unsigned):
    name = "Gaussian process regression"
    identifier = "gpfit"
    labels = ["misc", "unsigned", "unordered", "normal", "nonlinear", "directed"]

    def __init__(self, kernel="RBF"):
        self.identifier += f"_{kernel}"
        self._kernel = kernels.ConstantKernel() + kernels.WhiteKernel()
        self._kernel += getattr(kernels, kernel)()

    @parse_multivariate
    def multivariate(self, data):
        def _get_gpfit(pair, mat):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(kernel=self._kernel).fit(mat[pair[0]], np.ravel(mat[pair[1]]))
            y_predict = gp.predict(mat[pair[0]])
            return pair[0], pair[1], mean_squared_error(y_predict, np.ravel(mat[pair[1]]))
        
        pres = Parallel(n_jobs=-1)(delayed(_get_gpfit)(pair, data.to_numpy()) for pair in tqdm(permutations(range(data.n_processes), 2)))

        res_mat = np.zeros((data.n_processes, data.n_processes))
        for p in pres:
            res_mat[p[0], p[1]] = p[2]
        res_mat += res_mat.T

        return res_mat

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=self._kernel).fit(z[i], np.ravel(z[j]))
        y_predict = gp.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))


class PowerEnvelopeCorrelation(Undirected, Unsigned):
    humanname = "Power envelope correlation"
    identifier = "pec"
    labels = ["unsigned", "misc", "undirected"]

    def __init__(self, orth=False, log=False, absolute=False):
        self._orth = False
        if orth:
            self._orth = "pairwise"
            self.identifier += "_orth"
        self._log = log
        if log:
            self.identifier += "_log"
        self._absolute = absolute
        if absolute:
            self.identifier += "_abs"

    @parse_multivariate
    def multivariate(self, data):
        z = np.moveaxis(data.to_numpy(), 2, 0)
        con = mnec.envelope_correlation(
            z, orthogonalize=self._orth, log=self._log, absolute=self._absolute
            )
        print(con)
        adj = np.squeeze(con.get_data("dense"))
        np.fill_diagonal(adj, np.nan)
        return adj
