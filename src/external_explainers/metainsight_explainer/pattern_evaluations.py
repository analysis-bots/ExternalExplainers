from enum import Enum
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from diptest import diptest
from scipy.stats import gaussian_kde, zscore
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN


class PatternType(Enum):
    """
    An enumeration of the types of patterns.
    """
    NONE = 0
    OTHER = 1
    UNIMODALITY = 2
    TREND = 3
    OUTLIER = 4


class PatternEvaluator:
    """
    A class to evaluate different patterns in a series.
    """

    OUTLIER_ZSCORE_THRESHOLD = 2.0  # Z-score threshold for outlier detection
    TREND_SLOPE_THRESHOLD = 0.01  # Minimum absolute slope for trend detection

    @staticmethod
    def unimodality(series: pd.Series) -> (bool, Tuple[str, str]):
        """
        Evaluates if the series is unimodal using Hartigan's Dip test and returns the highlight.
        :param series: The series to evaluate.
        :return: (is_unimodal, highlight)
        """
        if isinstance(series, pd.Series):
            series = series.sort_values()
        else:
            return False, (None, None)
        vals = series.values
        if len(vals) < 4:
            return False, (None, None)
        # Perform Hartigan's Dip test
        dip_statistic, p_value = diptest(vals)
        is_unimodal = p_value > 0.05
        if not is_unimodal:
            return False, (None, None)
        # If there is unimodality, find the valley / peak
        # If a series is all 0s, then this can happen
        try:
            kde = gaussian_kde(series)
        except np.linalg.LinAlgError:
            return False, (None, None)

        # Evaluate the KDE over a range of values
        # Create a range of points covering the data span
        x_range = np.linspace(series.min(), series.max(), 1000)
        density_values = kde(x_range)

        # Find the index of the maximum (peak) and minimum (valley) density
        peak_index = np.argmax(density_values)
        valley_index = np.argmin(density_values)

        # Get the location of the peak / valley
        peak_location = x_range[peak_index]
        valley_location = x_range[valley_index]

        # Get the index from the real series for which the peak and valley occurr.
        # Because we are approximating, we get the index for which the values are the closest.
        peak_dist = np.inf
        valley_dist = np.inf
        valley_index = None
        peak_index = None
        for idx in series.index.tolist():
            val = series[idx]
            val_peak_dist = abs(val - peak_location)
            val_valley_dist = abs(val - valley_location)
            if val_peak_dist < peak_dist:
                peak_index = idx
                peak_dist = val_peak_dist
            if val_valley_dist < valley_dist:
                valley_index = idx
                valley_dist = val_valley_dist


        # Check which of the two is the bigger outlier, and return the one that is
        # furthest from the mean
        if abs(peak_location - series.mean()) > abs(valley_location - series.mean()):
            return True, (peak_index, 'Peak')
        else:
            return True, (valley_index, 'Valley')

    @staticmethod
    def trend(series: pd.Series) -> (bool, Tuple[str, str]):
        """
            Evaluates if a time series exhibits a significant trend (upward or downward).
            Uses linear regression to find the slope.
            Returns (True, highlight) if a trend is detected, (False, None) otherwise.
            Highlight is (slope, 'Upward' or 'Downward').
            """
        if len(series) < 2:
            return False, (None, None)

        # Check if the series is a time series, or just a series of numbers
        # We say a series is a time series if its index is either a datetime index or an increasing integer index
        is_datetime_index = isinstance(series.index, pd.DatetimeIndex)
        is_numeric_index = np.issubdtype(series.index.dtype, np.number)
        if is_numeric_index:
            series = series.sort_index()
            # Check if the index is strictly increasing
            is_increasing = np.all(np.diff(series.index) > 0)
        else:
            is_increasing = False

        # We can't find trends in series that are not time series -
        if not is_datetime_index and not is_increasing:
            return False, (None, None)

        # Create a simple linear model
        X = np.arange(len(series)).reshape(-1, 1)  # Independent variable (time index)
        y = series.values  # Dependent variable (data values)

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]

        # Check if the slope is significant
        if abs(slope) > PatternEvaluator.TREND_SLOPE_THRESHOLD:
            trend_direction = 'Upward' if slope > 0 else 'Downward'
            return True, (None, trend_direction)
        else:
            return False, (None, None)

    @staticmethod
    def outlier(series: pd.Series) -> (bool, Tuple[str, str]):
        """
        Evaluates if a series contains significant outliers.
        Uses the Z-score method.
        Returns (True, highlight) if outliers are detected, (False, None) otherwise.
        Highlight is a list of indices of the outlier points.
        """
        if len(series) < 2:
            return False, (None, None)

        # Calculate Z-scores
        z_scores = np.abs(zscore(series.dropna()))

        # Find indices where Z-score exceeds the threshold
        outlier_indices = np.where(z_scores > PatternEvaluator.OUTLIER_ZSCORE_THRESHOLD)[0]

        if len(outlier_indices) > 0:
            outlier_data_points = series.iloc[outlier_indices].values.tolist()
            outlier_index = series.index[outlier_indices].tolist()
            # If there are multiple outliers, use clustering and return the cluster means.
            # This is more informative and easier to interpret than a list of raw outlier values.
            if len(outlier_data_points) > 1:
                # Reshape for clustering
                outlier_data_points = np.array(outlier_data_points).reshape(-1, 1)
                # Perform clustering
                clustered = DBSCAN().fit_predict(outlier_data_points)
                cluster_means = []
                cluster_indexes = []
                for cluster in np.unique(clustered):
                    if cluster != -1:
                        cluster_points = outlier_data_points[clustered == cluster]
                        cluster_mean = np.mean(cluster_points)
                        cluster_means.append(cluster_mean)
                        # Take the most common index of the cluster points to represent the cluster
                        cluster_index = outlier_index[clustered == cluster]
                        cluster_index = pd.Series(cluster_index).mode()[0]
                        cluster_indexes.append(cluster_index)
                # If there are noise points, they will be labeled as -1 in DBSCAN. To us though, those are
                # not noise points, but outliers. So we will return them as well (unlike the clustered points,
                # their mean may be meaningless because they might be very far apart.
                noise_points = outlier_data_points[clustered == -1]
                if len(noise_points) > 0:
                    noise_points = noise_points.flatten().tolist()
                    cluster_means.extend(noise_points)
                # Return the cluster centers as the highlight meaning "outliers around these values"
                return True, (cluster_indexes, None)

            return True, ([outlier_index[0]], None)
        else:
            return False, (None, None)

    def __call__(self, series: pd.Series, pattern_type: PatternType) -> (bool, str):
        """
        Calls the appropriate pattern evaluation method based on the pattern type.
        :param series: The series to evaluate.
        :param pattern_type: The type of the pattern to evaluate.
        :return: (is_valid, highlight)
        """
        if pattern_type == PatternType.UNIMODALITY:
            return self.unimodality(series)
        elif pattern_type == PatternType.TREND:
            return self.trend(series)
        elif pattern_type == PatternType.OUTLIER:
            return self.outlier(series)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
