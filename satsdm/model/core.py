import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import multiprocessing as mp
import warnings
import os

from typing import List, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from .utils import (
    validate_gpd,
    crs_match,
    make_band_labels,
    repeat_array,
    validate_feature_types,
    validate_boolean,
    validate_numeric_scalar,
    to_iterable,
    create_output_raster_profile,
    get_raster_band_indexes,
    check_raster_alignment,
    NoDataException
)

Vector = Union[gpd.GeoSeries, gpd.GeoDataFrame]
ArrayLike = Union[np.ndarray, pd.DataFrame]
NCPUS = mp.cpu_count()

class MaxentConfig:
    # Set class variables (static variables) instead of instance variables
    clamp: bool = True
    beta_multiplier: float = 1.5
    beta_hinge: float = 1.0
    beta_lqp: float = 1.0
    beta_threshold: float = 1.0
    beta_categorical: float = 1.0
    feature_types: list = ["linear", "hinge", "product"]
    n_hinge_features: int = 10
    n_threshold_features: int = 10
    scorer: str = "roc_auc"
    tau: float = 0.5
    transform: str = "cloglog"
    tolerance: float = 2e-6
    use_lambdas: str = "best"
    n_lambdas: int = 100
    class_weights: Union[str, float] = 100

    @classmethod
    def update_feature_types(cls, feature_types):
        """Update feature types based on the selected checkboxes"""
        cls.feature_types = feature_types

    @classmethod
    def update_regularization(cls, beta_multiplier, beta_hinge, beta_lqp, beta_threshold, beta_categorical):
        """Update regularization parameters"""
        cls.beta_multiplier = beta_multiplier
        cls.beta_hinge = beta_hinge
        cls.beta_lqp = beta_lqp
        cls.beta_threshold = beta_threshold
        cls.beta_categorical = beta_categorical

    @classmethod
    def update_advanced_options(cls, clamp, tau, transform, tolerance, use_lambdas, n_lambdas, class_weights):
        """Update advanced options"""
        cls.clamp = clamp
        cls.tau = tau
        cls.transform = transform
        cls.tolerance = tolerance
        cls.use_lambdas = use_lambdas
        cls.n_lambdas = n_lambdas
        cls.class_weights = class_weights

def stack_geodataframes(
    presence: Vector,
    background: Vector,
    add_class_label: bool = False,
    target_crs: str = "presence"
) -> gpd.GeoDataFrame:
    """Concatenate geometries from two GeoSeries/GeoDataFrames."""
    validate_gpd(presence)
    validate_gpd(background)

    if isinstance(presence, gpd.GeoSeries):
        presence = presence.to_frame("geometry")
    if isinstance(background, gpd.GeoSeries):
        background = background.to_frame("geometry")

    crs = presence.crs
    if crs_match(presence.crs, background.crs):
        background.crs = presence.crs
    else:
        if target_crs.lower() == "presence":
            background.to_crs(crs, inplace=True)
        elif target_crs.lower() == "background":
            crs = background.crs
            presence.to_crs(crs, inplace=True)
        else:
            raise NameError(f"Unrecognized target_crs option: {target_crs}")

    if add_class_label:
        presence["class"] = 1
        background["class"] = 0

    matching = [col for col in presence.columns if col in background.columns]
    assert len(matching) > 0, "no matching columns found between data frames"

    merged = pd.concat((presence[matching], background[matching]), axis=0, ignore_index=True)
    return gpd.GeoDataFrame(merged, crs=crs)

class FeaturesMixin:
    """Methods for formatting x data and labels."""

    def _format_covariate_data(self, x: ArrayLike) -> Tuple[np.array, np.array]:
        """Reads input x data and formats it to consistent array dtypes."""
        if isinstance(x, np.ndarray):
            if self.categorical_ is None:
                con = x
                cat = None
            else:
                con = x[:, self.continuous_]
                cat = x[:, self.categorical_]

        elif isinstance(x, pd.DataFrame):
            con = x[self.continuous_pd_].to_numpy()
            cat = x[self.categorical_pd_].to_numpy() if len(self.categorical_pd_) > 0 else None

        else:
            raise TypeError(f"Unsupported x dtype: {type(x)}. Must be pd.DataFrame or np.array")

        return con, cat

    def _format_labels_and_dtypes(self, x: ArrayLike, categorical: list = None, labels: list = None) -> None:
        """Formats and stores label and type info from x for future indexing."""
        if isinstance(x, np.ndarray):
            nrows, ncols = x.shape
            continuous = list(range(ncols)) if categorical is None else list(set(range(ncols)) - set(categorical))
            self.labels_ = labels or make_band_labels(ncols)
            self.categorical_ = categorical
            self.continuous_ = continuous

        elif isinstance(x, pd.DataFrame):
            x.drop(columns=["geometry"], errors="ignore", inplace=True)
            self.labels_ = labels or list(x.columns)

            self.continuous_pd_ = list(x.select_dtypes(exclude="category").columns)
            self.categorical_pd_ = list(x.select_dtypes(include="category").columns)

            all_columns = list(x.columns)
            self.continuous_ = [all_columns.index(col) for col in self.continuous_pd_ if col in all_columns]
            self.categorical_ = [all_columns.index(col) for col in self.categorical_pd_ if col in all_columns] or None

        else:
            raise TypeError(f"Unsupported x dtype: {type(x)}")

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """Applies one-hot encoding to categorical covariate datasets."""

    def __init__(self):
        self.estimators_ = None

    def fit(self, x: ArrayLike) -> "CategoricalTransformer":
        self.estimators_ = []
        x = np.array(x)
        if x.ndim == 1:
            estimator = OneHotEncoder(dtype=np.uint8, sparse_output=False)
            self.estimators_.append(estimator.fit(x.reshape(-1, 1)))
        else:
            for col in range(x.shape[1]):
                xsub = x[:, col].reshape(-1, 1)
                estimator = OneHotEncoder(dtype=np.uint8, sparse_output=False)
                self.estimators_.append(estimator.fit(xsub))
        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        x = np.array(x)
        if x.ndim == 1:
            return self.estimators_[0].transform(x.reshape(-1, 1))
        else:
            class_data = [self.estimators_[col].transform(x[:, col].reshape(-1, 1))
                          for col in range(x.shape[1])]
            return np.concatenate(class_data, axis=1)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        return self.fit(x).transform(x)

def column_product(array: np.ndarray) -> np.ndarray:
    """Computes the column-wise product of a 2D array.

    Args:
        array: array-like of shape (n_samples, n_features)

    Returns:
        ndarray with of shape (n_samples, factorial(n_features-1))
    """
    nrows, ncols = array.shape

    if ncols == 1:
        return array
    else:
        products = []
        for xstart in range(0, ncols - 1):
            products.append(array[:, xstart].reshape(nrows, 1) * array[:, xstart + 1 :])
        return np.concatenate(products, axis=1)

def left_hinge(x: ArrayLike, mn: float, mx: float) -> np.ndarray:
    """Computes hinge transformation values.

    Args:
        x: Array-like of covariate values
        mn: Minimum covariate value to fit hinges to
        mx: Maximum covariate value to fit hinges to

    Returns:
        Array of hinge features
    """
    return np.minimum(1, np.maximum(0, (x - mn) / (repeat_array(mx, mn.shape[-1], axis=1) - mn)))

def right_hinge(x: ArrayLike, mn: float, mx: float) -> np.ndarray:
    """Computes hinge transformation values.

    Args:
        x: Array-like of covariate values
        mn: Minimum covariate value to fit hinges to
        mx: Maximum covariate value to fit hinges to

    Returns:
        Array of hinge features
    """
    mn_broadcast = repeat_array(mn, mx.shape[-1], axis=1)
    return np.minimum(1, np.maximum(0, (x - mn_broadcast) / (mx - mn_broadcast)))

class LinearTransformer(MinMaxScaler):
    """Applies linear feature transformations to rescale features from 0-1."""

    def __init__(
        self,
        clamp: bool = MaxentConfig.clamp,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.clamp = clamp
        self.feature_range = feature_range
        super().__init__(clip=clamp, feature_range=feature_range)

class ProductTransformer(BaseEstimator, TransformerMixin):
    """Computes the column-wise product of an array of input features, rescaling from 0-1."""

    def __init__(
        self,
        clamp: bool = MaxentConfig.clamp,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.clamp = clamp
        self.feature_range = feature_range
        self.estimator = None

    def fit(self, x: ArrayLike) -> "ProductTransformer":
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.

        Returns:
            self. Returns the transformer with fitted parameters.
        """
        self.estimator = MinMaxScaler(clip=self.clamp, feature_range=self.feature_range)
        self.estimator.fit(column_product(np.array(x)))

        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
        return self.estimator.transform(column_product(np.array(x)))

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x)
        return self.transform(x)

class HingeTransformer(BaseEstimator, TransformerMixin):
    """Fits hinge transformations to an array of covariates."""

    def __init__(self, n_hinges: int = MaxentConfig.n_hinge_features):
        self.n_hinges = n_hinges
        self.mins_ = None
        self.maxs_ = None
        self.hinge_indices_ = None

    def fit(self, x: ArrayLike) -> "HingeTransformer":
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.

        Returns:
            self. Updatesd transformer with fitted parameters.
        """
        x = np.array(x)
        self.mins_ = x.min(axis=0)
        self.maxs_ = x.max(axis=0)
        self.hinge_indices_ = np.linspace(self.mins_, self.maxs_, self.n_hinges)

        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
        x = np.array(x)
        xarr = repeat_array(x, self.n_hinges - 1, axis=-1)
        lharr = repeat_array(self.hinge_indices_[:-1].transpose(), len(x), axis=0)
        rharr = repeat_array(self.hinge_indices_[1:].transpose(), len(x), axis=0)
        lh = left_hinge(xarr, lharr, self.maxs_)
        rh = right_hinge(xarr, self.mins_, rharr)
        return np.concatenate((lh, rh), axis=2).reshape(x.shape[0], -1)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x)
        return self.transform(x)

class MaxentFeatureTransformer(BaseEstimator, TransformerMixin, FeaturesMixin):
    """Transforms covariate data into maxent-format feature data."""

    def __init__(
        self,
        feature_types: Union[str, list] = MaxentConfig.feature_types,
        clamp: bool = MaxentConfig.clamp,
        n_hinge_features: int = MaxentConfig.n_hinge_features,
        n_threshold_features: int = MaxentConfig.n_threshold_features,
    ):
        """Computes features based on the maxent feature types specified (like linear, quadratic, hinge).

        Args:
            feature_types: list of maxent features to generate.
            clamp: set feature values to global mins/maxs during prediction
            n_hinge_features: number of hinge knots to generate
            n_threshold_features: nuber of threshold features to generate
        """
        self.feature_types = feature_types
        self.clamp = clamp
        self.n_hinge_features = n_hinge_features
        self.n_threshold_features = n_threshold_features
        self.categorical_ = None
        self.continuous_ = None
        self.categorical_pd_ = None
        self.continuous_pd_ = None
        self.labels_ = None
        self.feature_names_ = None
        self.estimators_ = {
            "linear": None,
            "quadratic": None,
            "product": None,
            "threshold": None,
            "hinge": None,
            "categorical": None,
        }

    def fit(self, x: ArrayLike, categorical: list = None, labels: list = None) -> "MaxentFeatureTransformer":
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.
            categorical: indices indicating which x columns are categorical
            labels: covariate column labels. ignored if x is a pandas DataFrame

        Returns:
            self. Returns the transformer with fitted parameters.
        """
        self.feature_types = validate_feature_types(self.feature_types)
        self.clamp = validate_boolean(self.clamp)
        self.n_hinge_features = validate_numeric_scalar(self.n_hinge_features)
        self.n_threshold_features = validate_numeric_scalar(self.n_threshold_features)

        self._format_labels_and_dtypes(x, categorical=categorical, labels=labels)
        con, cat = self._format_covariate_data(x)
        nrows, ncols = con.shape

        feature_names = []
        if "linear" in self.feature_types:
            estimator = LinearTransformer(clamp=self.clamp)
            estimator.fit(con)
            self.estimators_["linear"] = estimator
            feature_names += ["linear"] * estimator.n_features_in_

        if "quadratic" in self.feature_types:
            estimator = QuadraticTransformer(clamp=self.clamp)
            estimator.fit(con)
            self.estimators_["quadratic"] = estimator
            feature_names += ["quadratic"] * estimator.estimator.n_features_in_

        if "product" in self.feature_types:
            estimator = ProductTransformer(clamp=self.clamp)
            estimator.fit(con)
            self.estimators_["product"] = estimator
            feature_names += ["product"] * estimator.estimator.n_features_in_

        if "threshold" in self.feature_types:
            estimator = ThresholdTransformer(n_thresholds=self.n_threshold_features)
            estimator.fit(con)
            self.estimators_["threshold"] = estimator
            feature_names += ["threshold"] * (estimator.n_thresholds * ncols)

        if "hinge" in self.feature_types:
            estimator = HingeTransformer(n_hinges=self.n_hinge_features)
            estimator.fit(con)
            self.estimators_["hinge"] = estimator
            feature_names += ["hinge"] * ((estimator.n_hinges - 1) * 2 * ncols)

        if cat is not None:
            estimator = CategoricalTransformer()
            estimator.fit(cat)
            self.estimators_["categorical"] = estimator
            for est in estimator.estimators_:
                feature_names += ["categorical"] * len(est.categories_[0])

        self.feature_names_ = feature_names

        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
        con, cat = self._format_covariate_data(x)
        features = []

        if "linear" in self.feature_types:
            features.append(self.estimators_["linear"].transform(con))

        if "quadratic" in self.feature_types:
            features.append(self.estimators_["quadratic"].transform(con))

        if "product" in self.feature_types:
            features.append(self.estimators_["product"].transform(con))

        if "threshold" in self.feature_types:
            features.append(self.estimators_["threshold"].transform(con))

        if "hinge" in self.feature_types:
            features.append(self.estimators_["hinge"].transform(con))

        if cat is not None:
            features.append(self.estimators_["categorical"].transform(cat))

        return np.concatenate(features, axis=1)

    def fit_transform(self, x: ArrayLike, categorical: list = None, labels: list = None) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x, categorical=categorical, labels=labels)
        return self.transform(x)

class SDMMixin:
    """Mixin class for SDM classifiers."""

    _estimator_type = "classifier"
    classes_ = [0, 1]

    def score(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None) -> float:
        """Return the mean AUC score on the given test data and labels.

        Args:
            x: test samples. array-like of shape (n_samples, n_features).
            y: presence/absence labels. array-like of shape (n_samples,).
            sample_weight: array-like of shape (n_samples,)

        Returns:
            AUC score of `self.predict(x)` w.r.t. `y`.
        """
        return roc_auc_score(y, self.predict(x), sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_y": True}

    def permutation_importance_scores(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        n_repeats: int = 10,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Compute a generic feature importance score by modifying feature values
            and computing the relative change in model performance.

        Permutation importance measures how much a model score decreases when a
            single feature value is randomly shuffled. This score doesn't reflect
            the intrinsic predictive value of a feature by itself, but how important
            feature is for a particular model.

        Args:
            x: test samples. array-like of shape (n_samples, n_features).
            y: presence/absence labels. array-like of shape (n_samples,).
            sample_weight: array-like of shape (n_samples,)
            n_repeats: number of permutation iterations.
            n_jobs: number of parallel compute tasks. set to -1 for all cpus.

        Returns:
            importances: an array of shape (n_features, n_repeats).
        """
        pi = permutation_importance(self, x, y, sample_weight=sample_weight, n_jobs=n_jobs, n_repeats=n_repeats)

        return pi.importances

    def permutation_importance_plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        n_repeats: int = 10,
        labels: list = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a box plot with bootstrapped permutation importance scores for each covariate.

        Permutation importance measures how much a model score decreases when a
            single feature value is randomly shuffled. This score doesn't reflect
            the intrinsic predictive value of a feature by itself, but how important
            feature is for a particular model.

        It is often appropriate to compute permuation importance scores using both
            training and validation sets. Large differences between the two may
            indicate overfitting.

        This implementation does not necessarily match the implementation in Maxent.
            These scores may be difficult to interpret if there is a high degree
            of covariance between features or if the model estimator includes any
            non-linear feature transformations (e.g. 'hinge' features).

        Reference:
            https://scikit-learn.org/stable/modules/permutation_importance.html

        Args:
            x: evaluation features. array-like of shape (n_samples, n_features).
            y: presence/absence labels. array-like of shape (n_samples,).
            sample_weight: array-like of shape (n_samples,)
            n_repeats: number of permutation iterations.
            labels: list of band names to label the plots.
            **kwargs: additional arguments to pass to `plt.subplots()`.

        Returns:
            fig, ax: matplotlib subplot figure and axes.
        """
        importance = self.permutation_importance_scores(x, y, sample_weight=sample_weight, n_repeats=n_repeats)
        rank_order = importance.mean(axis=-1).argsort()

        if labels is None:
            try:
                labels = x.columns.tolist()
            except AttributeError:
                labels = make_band_labels(x.shape[-1])
        labels = [labels[idx] for idx in rank_order]

        plot_defaults = {"dpi": 150, "figsize": (5, 4)}
        plot_defaults.update(**kwargs)
        fig, ax = plt.subplots(**plot_defaults)
        ax.boxplot(
            importance[rank_order].T,
            vert=False,
            labels=labels,
        )
        fig.tight_layout()

        return fig, ax

    def partial_dependence_scores(
        self,
        x: ArrayLike,
        percentiles: tuple = (0.025, 0.975),
        n_bins: int = 100,
        categorical_features: tuple = [None],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute partial dependence scores for each feature.

        Args:
            x: evaluation features. array-like of shape (n_samples, n_features).
                used to constrain the range of values to evaluate for each feature.
            percentiles: lower and upper percentiles used to set the range to plot.
            n_bins: the number of bins spanning the lower-upper percentile range.
            categorical_features: a 0-based index of which features are categorical.

        Returns:
            bins, mean, stdv: the binned feature values and the mean/stdv of responses.
        """
        ncols = x.shape[1]
        mean = np.zeros((ncols, n_bins))
        stdv = np.zeros_like(mean)
        bins = np.zeros_like(mean)

        for idx in range(ncols):
            if idx in categorical_features:
                continue
            pd = partial_dependence(
                self,
                x,
                [idx],
                percentiles=percentiles,
                grid_resolution=n_bins,
                kind="individual",
            )
            mean[idx] = pd["individual"][0].mean(axis=0)
            stdv[idx] = pd["individual"][0].std(axis=0)
            bins[idx] = pd["grid_values"][0]

        return bins, mean, stdv

    def partial_dependence_plot(
        self,
        x: ArrayLike,
        percentiles: tuple = (0.025, 0.975),
        n_bins: int = 50,
        categorical_features: tuple = None,
        labels: list = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the response of an estimator across the range of feature values.

        Args:
            x: evaluation features. array-like of shape (n_samples, n_features).
                used to constrain the range of values to evaluate for each feature.
            percentiles: lower and upper percentiles used to set the range to plot.
            n_bins: the number of bins spanning the lower-upper percentile range.
            categorical_features: a 0-based index of which features are categorical.
            labels: list of band names to label the plots.
            **kwargs: additional arguments to pass to `plt.subplots()`.

        Returns:
            fig, ax: matplotlib subplot figure and axes.
        """
        # skip categorical features for now
        if categorical_features is None:
            try:
                categorical_features = self.transformer.categorical_ or [None]
            except AttributeError:
                categorical_features = [None]

        bins, mean, stdv = self.partial_dependence_scores(
            x, percentiles=percentiles, n_bins=n_bins, categorical_features=categorical_features
        )

        if labels is None:
            try:
                labels = x.columns.tolist()
            except AttributeError:
                labels = make_band_labels(x.shape[-1])

        ncols = x.shape[1]
        figx = int(np.ceil(np.sqrt(ncols)))
        figy = int(np.ceil(ncols / figx))
        fig, ax = plt.subplots(figx, figy, **kwargs)
        ax = ax.flatten()

        for idx in range(ncols):
            ax[idx].fill_between(bins[idx], mean[idx] - stdv[idx], mean[idx] + stdv[idx], alpha=0.25)
            ax[idx].plot(bins[idx], mean[idx])
            ax[idx].set_title(labels[idx])

        # turn off empty plots
        for axi in ax:
            if not axi.lines:
                axi.set_visible(False)

        fig.tight_layout()

        return fig, ax

def compute_lambdas(
    y: ArrayLike, weights: ArrayLike, reg: ArrayLike, n_lambdas: int = MaxentConfig.n_lambdas
) -> np.ndarray:
    """Computes lambda parameter values for elastic lasso fits.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        weights: per-sample model weights
        reg: per-feature regularization coefficients
        n_lambdas: number of lambda values to estimate

    Returns:
        lambdas: Array of lambda scores of length n_lambda
    """
    n_presence = np.sum(y)
    mean_regularization = np.mean(reg)
    total_weight = np.sum(weights)
    seed_range = np.linspace(4, 0, n_lambdas)
    lambdas = 10 ** (seed_range) * mean_regularization * (n_presence / total_weight)

    return lambdas

def compute_regularization(
    y: ArrayLike,
    z: np.ndarray,
    feature_labels: List[str],
    beta_multiplier: float = MaxentConfig.beta_multiplier,
    beta_lqp: float = MaxentConfig.beta_lqp,
    beta_threshold: float = MaxentConfig.beta_threshold,
    beta_hinge: float = MaxentConfig.beta_hinge,
    beta_categorical: float = MaxentConfig.beta_hinge,
) -> np.ndarray:
    """Computes variable regularization values for all feature data.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        z: model features (transformations applied to covariates)
        feature_labels: list of length n_features, with labels identifying each column's feature type
            with options ["linear", "quadratic", "product", "threshold", "hinge", "categorical"]
        beta_multiplier: scaler for all regularization parameters. higher values exclude more features
        beta_lqp: scaler for linear, quadratic and product feature regularization
        beta_threshold: scaler for threshold feature regularization
        beta_hinge: scaler for hinge feature regularization
        beta_categorical: scaler for categorical feature regularization

    Returns:
        max_reg: Array with per-feature regularization parameters
    """
    # compute regularization based on presence-only locations
    z1 = z[y == 1]
    nrows, ncols = z1.shape
    labels = np.array(feature_labels)
    nlabels = len(feature_labels)

    assert nlabels == ncols, f"number of feature_labels ({nlabels}) must match number of features ({ncols})"

    # create arrays to store the regularization params
    base_regularization = np.zeros(ncols)
    hinge_regularization = np.zeros(ncols)
    threshold_regularization = np.zeros(ncols)

    # use a different reg table based on the features set
    if "product" in labels:
        table_lqp = RegularizationConfig.product
    elif "quadratic" in labels:
        table_lqp = RegularizationConfig.quadratic
    else:
        table_lqp = RegularizationConfig.linear

    if "linear" in labels:
        linear_idxs = labels == "linear"
        fr_max, fr_min = table_lqp
        multiplier = beta_lqp
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[linear_idxs] = reg

    if "quadratic" in labels:
        quadratic_idxs = labels == "quadratic"
        fr_max, fr_min = table_lqp
        multiplier = beta_lqp
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[quadratic_idxs] = reg

    if "product" in labels:
        product_idxs = labels == "product"
        fr_max, fr_min = table_lqp
        multiplier = beta_lqp
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[product_idxs] = reg

    if "threshold" in labels:
        threshold_idxs = labels == "threshold"
        fr_max, fr_min = RegularizationConfig.threshold
        multiplier = beta_threshold
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[threshold_idxs] = reg

        # increase regularization for uniform threshlold values
        all_zeros = np.all(z1 == 0, axis=0)
        all_ones = np.all(z1 == 1, axis=0)
        threshold_regularization[all_zeros] = 1
        threshold_regularization[all_ones] = 1

    if "hinge" in labels:
        hinge_idxs = labels == "hinge"
        fr_max, fr_min = RegularizationConfig.hinge
        multiplier = beta_hinge
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[hinge_idxs] = reg

        # increase regularization for extreme hinge values
        hinge_std = np.std(z1[:, hinge_idxs], ddof=1, axis=0)
        hinge_sqrt = np.zeros(len(hinge_std)) + (1 / np.sqrt(nrows))
        std = np.max((hinge_std, hinge_sqrt), axis=0)
        hinge_regularization[hinge_idxs] = (0.5 * std) / np.sqrt(nrows)

    if "categorical" in labels:
        categorical_idxs = labels == "categorical"
        fr_max, fr_min = RegularizationConfig.categorical
        multiplier = beta_categorical
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[categorical_idxs] = reg

    # compute the maximum regularization based on a few different approaches
    default_regularization = 0.001 * (np.max(z, axis=0) - np.min(z, axis=0))
    variance_regularization = np.std(z1, ddof=1, axis=0) * base_regularization
    max_regularization = np.max(
        (default_regularization, variance_regularization, hinge_regularization, threshold_regularization), axis=0
    )

    # apply the final scaling factor
    max_regularization *= beta_multiplier

    return max_regularization

def compute_weights(y: ArrayLike, pbr: int = 100) -> np.ndarray:
    """Compute Maxent-format per-sample model weights.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        pbr: presence-to-background weight ratio. pbr=100 sets background samples to 1/100 weight of presence samples.

    Returns:
        weights: array with glmnet-formatted sample weights
    """
    weights = np.array(y + (1 - y) * pbr)
    return weights

def format_occurrence_data(y: ArrayLike) -> ArrayLike:
    """Reads input y data and formats it to consistent 1d array dtypes.

    Args:
        y: array-like of shape (n_samples,) or (n_samples, 1)

    Returns:
        formatted uint8 ndarray of shape (n_samples,)

    Raises:
        np.AxisError: an array with 2 or more columns is passed
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if y.ndim > 1:
        if y.shape[1] > 1 or y.ndim > 2:
            raise np.AxisError(f"Multi-column y data passed of shape {y.shape}. Must be 1d or 1 column.")
        y = y.flatten()

    return y.astype("uint8")

def estimate_C_from_betas(beta_multiplier: float) -> float:
    """Convert the maxent-format beta_multiplier to an sklearn-format C regularization parameter.

    Args:
        beta_multiplier: the maxent beta regularization scaler

    Returns:
        a C factor approximating the level of regularization passed to glmnet
    """
    return 2 / (1 - np.exp(-beta_multiplier))

def maxent_raw_transform(engma: np.ndarray) -> np.ndarray:
    """Compute maxent's raw suitability score

    Args:
        engma: calibrated maxent linear model output

    Returns:
        the log-linear raw scores for each sample
    """
    return np.exp(engma)

def maxent_alpha(raw: np.ndarray) -> float:
    """Compute the sum-to-one alpha maxent model parameter.

    Args:
        raw: uncalibrated maxent raw (exponential) model output

    Returns:
        alpha: the output sum-to-one scaling factor
    """
    return -np.log(np.sum(raw))

def maxent_entropy(raw: np.ndarray) -> float:
    """Compute the maxent model entropy score for scaling the logistic output

    Args:
        raw: uncalibrated maxent raw (exponential) model output

    Returns:
        entropy: background distribution entropy score
    """
    scaled = raw / np.sum(raw)
    return -np.sum(scaled * np.log(scaled))

def maxent_cloglog_transform(engma: np.ndarray, entropy: float) -> np.ndarray:
    """Compute maxent's cumulative log-log suitability score

    Args:
        engma: calibrated maxent linear model output
        entropy: the calibrated model entropy score

    Returns:
        the cloglog scores for each sample
    """
    return 1 - np.exp(-np.exp(engma) * np.exp(entropy))

class MaxentModel(BaseEstimator, SDMMixin):
    """Model estimator for Maxent-style species distribution models."""

    def __init__(
        self,
        feature_types: Union[list, str] = MaxentConfig.feature_types,
        tau: float = MaxentConfig.tau,
        transform: float = MaxentConfig.transform,
        clamp: bool = MaxentConfig.clamp,
        scorer: str = MaxentConfig.scorer,
        beta_multiplier: float = MaxentConfig.beta_multiplier,
        beta_lqp: float = MaxentConfig.beta_lqp,
        beta_hinge: float = MaxentConfig.beta_hinge,
        beta_threshold: float = MaxentConfig.beta_threshold,
        beta_categorical: float = MaxentConfig.beta_categorical,
        n_hinge_features: int = MaxentConfig.n_hinge_features,
        n_threshold_features: int = MaxentConfig.n_threshold_features,
        convergence_tolerance: float = MaxentConfig.tolerance,
        use_lambdas: str = MaxentConfig.use_lambdas,
        n_lambdas: int = MaxentConfig.n_lambdas,
        class_weights: Union[str, float] = MaxentConfig.class_weights,
        n_cpus: int = NCPUS,
        use_sklearn: bool = True,
    ):
        """Create a maxent model object.

        Args:
            feature_types: maxent feature types to fit. must be in string "lqphta" or
                list ["linear", "quadratic", "product", "hinge", "threshold", "auto"]
            tau: maxent prevalence value for scaling logistic output
            transform: maxent model transformation type. select from
                ["raw", "logistic", "cloglog"].
            clamp: set features to min/max range from training during prediction
            scorer: sklearn scoring function for model training
            beta_multiplier: scaler for all regularization parameters.
                higher values drop more coeffiecients
            beta_lqp: linear, quadratic and product feature regularization scaler
            beta_hinge: hinge feature regularization scaler
            beta_threshold: threshold feature regularization scaler
            beta_categorical: categorical feature regularization scaler
            n_hinge_features: the number of hinge features to fit in feature transformation
            n_threshold_features: the number of thresholds to fit in feature transformation
            convergence_tolerance: model convergence tolerance level
            use_lambdas: guide for which model lambdas to select (either "best" or "last")
            n_lambdas: number of lamba values to fit models with
            class_weights: strategy for weighting presence samples.
                pass "balanced" to compute the ratio based on sample frequency
                or pass a float for the presence:background weight ratio
                the R `maxnet` package uses a value of 100 as default.
                set to None to ignore.
            n_cpus: threads to use during model training
            use_sklearn: force using `sklearn` for fitting logistic regression.
                turned off by default to use `glmnet` for fitting.
                this feature was turned on to support Windows users
                who install the package without a fortran compiler.
        """

        print("Clamp value:", clamp)
        
        self.feature_types = feature_types
        self.tau = tau
        self.transform = transform
        self.clamp = clamp
        self.scorer = scorer
        self.beta_multiplier = beta_multiplier
        self.beta_hinge = beta_hinge
        self.beta_lqp = beta_lqp
        self.beta_threshold = beta_threshold
        self.beta_categorical = beta_categorical
        self.n_hinge_features = n_hinge_features
        self.n_threshold_features = n_threshold_features
        self.convergence_tolerance = convergence_tolerance
        self.n_cpus = n_cpus
        self.use_lambdas = use_lambdas
        self.n_lambdas = n_lambdas
        self.class_weights = class_weights
        self.use_sklearn = use_sklearn

        # computed during model fitting
        self.initialized_ = False
        self.estimator = None
        self.preprocessor = None
        self.transformer = None
        self.regularization_ = None
        self.lambdas_ = None
        self.beta_scores_ = None
        self.entropy_ = 0.0
        self.alpha_ = 0.0

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        categorical: List[int] = None,
        labels: list = None,
        preprocessor: BaseEstimator = None,
    ) -> None:
        """Trains a maxent model using a set of covariates and presence/background points.

        Args:
            x: array of shape (n_samples, n_features) with covariate data
            y: array of shape (n_samples,) with binary presence/background (1/0) values
            sample_weight: array of weights assigned to each sample with shape (n_samples,).
                this is modified by the `class_weights` model parameter unless
                you set `class_weights=None`.
            categorical: indices for which columns are categorical
            labels: covariate labels. ignored if x is a pandas DataFrame
            preprocessor: an `sklearn` transformer with a .transform() and/or
                a .fit_transform() method. Some examples include a PCA() object or a
                RobustScaler().
        """
        # clear state variables
        self.alpha_ = 0.0
        self.entropy_ = 0.0

        # format the input data
        y = format_occurrence_data(y)

        # apply preprocessing
        if preprocessor is not None:
            self.preprocessor = preprocessor
            try:
                x = self.preprocessor.transform(x)
            except NotFittedError:
                x = self.preprocessor.fit_transform(x)

        # fit the feature transformer
        self.feature_types = validate_feature_types(self.feature_types)
        self.transformer = MaxentFeatureTransformer(
            feature_types=self.feature_types,
            clamp=self.clamp,
            n_hinge_features=self.n_hinge_features,
            n_threshold_features=self.n_threshold_features,
        )
        features = self.transformer.fit_transform(x, categorical=categorical, labels=labels)
        feature_labels = self.transformer.feature_names_

        # compute class weights
        if self.class_weights is not None:
            pbr = len(y) / y.sum() if self.class_weights == "balanced" else self.class_weights
            class_weight = compute_weights(y, pbr=pbr)

            # scale the sample weight
            if sample_weight is None:
                sample_weight = class_weight
            else:
                sample_weight *= class_weight

        # model fitting with sklearn
        if self.use_sklearn:
            C = estimate_C_from_betas(self.beta_multiplier)
            self.initialize_sklearn_model(C)
            self.estimator.fit(features, y, sample_weight=sample_weight)
            self.beta_scores_ = self.estimator.coef_[0]

        # model fitting with glmnet
        else:
            # set feature regularization parameters
            self.regularization_ = compute_regularization(
                y,
                features,
                feature_labels=feature_labels,
                beta_multiplier=self.beta_multiplier,
                beta_lqp=self.beta_lqp,
                beta_threshold=self.beta_threshold,
                beta_hinge=self.beta_hinge,
                beta_categorical=self.beta_categorical,
            )

            # get model lambda scores to initialize the glm
            self.lambdas_ = compute_lambdas(y, sample_weight, self.regularization_, n_lambdas=self.n_lambdas)

            # model fitting
            self.initialize_glmnet_model(lambdas=self.lambdas_)
            self.estimator.fit(
                features,
                y,
                sample_weight=sample_weight,
                relative_penalties=self.regularization_,
            )

            # get the beta values based on which lambda selection method to use
            if self.use_lambdas == "last":
                self.beta_scores_ = self.estimator.coef_path_[0, :, -1]
            elif self.use_lambdas == "best":
                self.beta_scores_ = self.estimator.coef_path_[0, :, self.estimator.lambda_max_inx_]

        # store initialization state
        self.initialized_ = True

        # apply maxent-specific transformations
        class_transform = self.get_params()["transform"]
        self.set_params(transform="raw")
        raw = self.predict(x[y == 0])
        self.set_params(transform=class_transform)

        # alpha is a normalizing constant that ensures that f1(z) integrates (sums) to 1
        self.alpha_ = maxent_alpha(raw)

        # the distance from f(z) is the relative entropy of f1(z) WRT f(z)
        self.entropy_ = maxent_entropy(raw)

        return self

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Apply a model to a set of covariates or features. Requires that a model has been fit.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

        Returns:
            predictions: array-like of shape (n_samples,) with model predictions
        """
        if not self.initialized_:
            raise NotFittedError("Model must be fit first")

        # feature transformations
        x = x if self.preprocessor is None else self.preprocessor.transform(x)
        features = x if self.transformer is None else self.transformer.transform(x)

        # apply the model
        engma = np.matmul(features, self.beta_scores_) + self.alpha_

        # scale based on the transform type
        if self.transform == "raw":
            return maxent_raw_transform(engma)

        elif self.transform == "logistic":
            return maxent_logistic_transform(engma, self.entropy_, self.tau)

        elif self.transform == "cloglog":
            return maxent_cloglog_transform(engma, self.entropy_)

    def predict_proba(self, x: ArrayLike) -> ArrayLike:
        """Compute prediction probability scores for the 0/1 classes.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

        Returns:
            predictions: array-like of shape (n_samples, 2) with model predictions
        """
        ypred = self.predict(x).reshape(-1, 1)
        predictions = np.hstack((1 - ypred, ypred))

        return predictions

    def fit_predict(
        self,
        x: ArrayLike,
        y: ArrayLike,
        categorical: list = None,
        labels: list = None,
        preprocessor: BaseEstimator = None,
    ) -> ArrayLike:
        """Trains and applies a model to x/y data.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: column indices indicating which columns are categorical
            labels: Covariate labels. Ignored if x is a pandas DataFrame
            preprocessor: an `sklearn` transformer with a .transform() and/or
                a .fit_transform() method. Some examples include a PCA() object or a
                RobustScaler().

        Returns:
            predictions: Array-like of shape (n_samples,) with model predictions
        """
        self.fit(x, y, categorical=categorical, labels=labels, preprocessor=preprocessor)
        predictions = self.predict(x)

        return predictions

    def initialize_glmnet_model(
        self,
        lambdas: np.array,
        alpha: float = 1,
        standardize: bool = False,
        fit_intercept: bool = True,
    ) -> None:
        """Creates the Logistic Regression with elastic net penalty model object.

        Args:
            lambdas: array of model lambda values. get from elapid.features.compute_lambdas()
            alpha: elasticnet mixing parameter. alpha=1 for lasso, alpha=0 for ridge
            standardize: specify coefficient normalization
            fit_intercept: include an intercept parameter
        """
        self.estimator = LogitNet(
            alpha=alpha,
            lambda_path=lambdas,
            standardize=standardize,
            fit_intercept=fit_intercept,
            scoring=self.scorer,
            n_jobs=self.n_cpus,
            tol=self.convergence_tolerance,
        )

    def initialize_sklearn_model(self, C: float, fit_intercept: bool = True) -> None:
        """Creates an sklearn Logisticregression estimator with L1 penalties.

        Args:
            C: the regularization parameter
            fit_intercept: include an intercept parameter
        """
        self.estimator = LogisticRegression(
            C=C,
            fit_intercept=fit_intercept,
            penalty="l1",
            solver="liblinear",
            tol=self.convergence_tolerance,
            max_iter=self.n_lambdas,
        )
    def jackknife_importance(self, X, y):
        """
        Perform jackknife variable importance analysis by
        evaluating performance with and without each variable.
        Returns a DataFrame with scores per variable.
        """
        from sklearn.metrics import roc_auc_score
        import numpy as np
        import pandas as pd

        base_score = roc_auc_score(y, self.predict(X))
        results = []

        for var in X.columns:
            # Only variable i
            x_only = X[[var]]
            self.fit(x_only, y)
            auc_only = roc_auc_score(y, self.predict(x_only))

            # Without variable i
            x_wo = X.drop(columns=[var])
            self.fit(x_wo, y)
            auc_wo = roc_auc_score(y, self.predict(x_wo))

            results.append({
                "Variable": var,
                "AUC (only)": auc_only,
                "AUC (without)": auc_wo,
                "Full model AUC": base_score
            })

        # Re-train on full set to restore model
        self.fit(X, y)
        return pd.DataFrame(results)

def apply_model_to_array(
    model: BaseEstimator,
    array: np.ndarray,
    nodata: float,
    nodata_idx: int,
    count: int = 1,
    dtype: str = "float32",
    predict_proba: bool = False,
    **kwargs,
) -> np.ndarray:
    """Applies a model to an array of covariates.

    Covariate array should be of shape (nbands, nrows, ncols).

    Args:
        model: object with a `model.predict()` function
        array: array of shape (nbands, nrows, ncols) with pixel values
        nodata: numeric nodata value to apply to the output array
        nodata_idx: array of bools with shape (nbands, nrows, ncols) containing nodata locations
        count: number of bands in the prediction output
        dtype: prediction array dtype
        predict_proba: use model.predict_proba() instead of model.predict()
        **kwargs: additonal keywords to pass to model.predict()

    Returns:
        ypred_window: Array of shape (nrows, ncols) with model predictions
    """
    # only apply to valid pixels
    valid = ~nodata_idx.any(axis=0)
    covariates = array[:, valid].transpose()
    ypred = model.predict(covariates, **kwargs) if not predict_proba else model.predict_proba(covariates, **kwargs)

    # reshape to the original window size
    rows, cols = valid.shape
    ypred_window = np.zeros((count, rows, cols), dtype=dtype) + nodata
    ypred_window[:, valid] = ypred.transpose()

    return ypred_window

def apply_model_to_rasters(
    model: BaseEstimator,
    raster_paths: list,
    output_path: str,
    resampling: rio.enums.Enum = rio.enums.Resampling.average,
    count: int = 1,
    dtype: str = "float32",
    nodata: float = -9999,
    driver: str = "GTiff",
    compress: str = "deflate",
    bigtiff: bool = True,
    template_idx: int = 0,
    windowed: bool = True,
    predict_proba: bool = False,
    ignore_sklearn: bool = True,
    quiet: bool = False,
    **kwargs,
) -> None:
    """Applies a trained model to a list of raster datasets.

    The list and band order of the rasters must match the order of the covariates
    used to train the model. It reads each dataset block-by-block, applies
    the model, and writes gridded predictions. If the raster datasets are not
    consistent (different extents, resolutions, etc.), it wll re-project the data
    on the fly, with the grid size, extent and projection based on a 'template'
    raster.

    Args:
        model: object with a model.predict() function
        raster_paths: raster paths of covariates to apply the model to
        output_path: path to the output file to create
        resampling: resampling algorithm to apply to on-the-fly reprojection
            from rasterio.enums.Resampling
        count: number of bands in the prediction output
        dtype: the output raster data type
        nodata: output nodata value
        driver: output raster format
            from rasterio.drivers.raster_driver_extensions()
        compress: compression to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)
        template_idx: index of the raster file to use as a template.
            template_idx=0 sets the first raster as template
        windowed: apply the model using windowed read/write
            slower, but more memory efficient
        predict_proba: use model.predict_proba() instead of model.predict()
        ignore_sklearn: silence sklearn warning messages
        quiet: silence progress bar output
        **kwargs: additonal keywords to pass to model.predict()

    Returns:
        None: saves model predictions to disk.
    """
    
    # make sure the raster_paths are iterable
    raster_paths = to_iterable(raster_paths)

    # get and set template parameters
    windows, dst_profile = create_output_raster_profile(
        raster_paths,
        template_idx,
        count=count,
        windowed=windowed,
        nodata=nodata,
        compress=compress,
        driver=driver,
        bigtiff=bigtiff,
    )

    # get the bands and indexes for each covariate raster
    nbands, band_idx = get_raster_band_indexes(raster_paths)

    # check whether the raster paths are aligned to determine how the data are read
    aligned = check_raster_alignment(raster_paths)

    # set a dummy nodata variable if none is set
    # (acutal nodata reads handled by rasterios src.read(masked=True) method)
    nodata = nodata or 0

    # turn off sklearn warnings
    if ignore_sklearn:
        warnings.filterwarnings("ignore", category=UserWarning)

    # open all rasters to read from later
    srcs = [rio.open(raster_path) for raster_path in raster_paths]

    # use warped VRT reads to align all rasters pixel-pixel if not aligned
    if not aligned:
        vrt_options = {
            "resampling": resampling,
            "transform": dst_profile["transform"],
            "crs": dst_profile["crs"],
            "height": dst_profile["height"],
            "width": dst_profile["width"],
        }
        srcs = [rio.vrt.WarpedVRT(src, **vrt_options) for src in srcs]

    with rio.open(output_path, "w", **dst_profile) as dst:
        total_windows = len(windows)
        for i, window in enumerate(windows):
            if not quiet:
                print(f"Processing window {i + 1} of {total_windows}")
            # create stacked arrays to handle multi-raster, multi-band inputs
            # that may have different nodata locations
            covariates = np.zeros((nbands, window.height, window.width), dtype=np.float32)
            nodata_idx = np.ones_like(covariates, dtype=bool)

            try:
                for i, src in enumerate(srcs):
                    data = src.read(window=window, masked=True)
                    covariates[band_idx[i] : band_idx[i + 1]] = data
                    nodata_idx[band_idx[i] : band_idx[i + 1]] = data.mask

                    # skip blocks full of no-data
                    if data.mask.all():
                        raise NoDataException()

                predictions = apply_model_to_array(
                    model,
                    covariates,
                    nodata,
                    nodata_idx,
                    count=count,
                    dtype=dtype,
                    predict_proba=predict_proba,
                    **kwargs,
                )
                dst.write(predictions, window=window)

            except NoDataException:
                continue
