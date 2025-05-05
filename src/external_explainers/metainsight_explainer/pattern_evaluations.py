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
        # Perform Hartigan's Dip test
        dip_statistic, p_value = diptest(series.dropna().values)
        is_unimodal = p_value > 0.05
        if not is_unimodal:
            return False, (None, None)
        # If there is unimodality, find the valley / peak
        # 2. Perform Kernel Density Estimation
        kde = gaussian_kde(series)

        # 3. Evaluate the KDE over a range of values
        # Create a range of points covering the data span
        x_range = np.linspace(series.min(), series.max(), 1000)
        density_values = kde(x_range)

        # 4. Find the index of the maximum (peak) and minimum (valley) density
        peak_index = np.argmax(density_values)
        valley_index = np.argmin(density_values)

        # 5. Map indices back to data values to get the estimated locations
        peak_location = x_range[peak_index]
        valley_location = x_range[valley_index]

        # Check which of the two is the bigger outlier, and return the one that is
        # furthest from the mean
        if abs(peak_location - series.mean()) > abs(valley_location - series.mean()):
            return True, (peak_location, 'Peak')
        else:
            return True, (valley_location, 'Valley')



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

        # Create a simple linear model
        X = np.arange(len(series)).reshape(-1, 1)  # Independent variable (time index)
        y = series.values  # Dependent variable (data values)

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]

        # Check if the slope is significant
        if abs(slope) > PatternEvaluator.TREND_SLOPE_THRESHOLD:
            trend_direction = 'Upward' if slope > 0 else 'Downward'
            return True, (slope, trend_direction)
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
            outlier_data_points = series[outlier_indices].values.tolist()
            # If there are multiple outliers, use clustering and return the cluster means.
            # This is more informative and easier to interpret than a list of raw outlier values.
            if len(outlier_data_points) > 1:
                # Reshape for clustering
                outlier_data_points = np.array(outlier_data_points).reshape(-1, 1)
                # Perform clustering
                clustered = DBSCAN().fit_predict(outlier_data_points)
                cluster_means = []
                for cluster in np.unique(clustered):
                    if cluster != -1:
                        cluster_points = outlier_data_points[clustered == cluster]
                        cluster_mean = np.mean(cluster_points)
                        cluster_means.append(cluster_mean)
                # If there are noise points, they will be labeled as -1 in DBSCAN. To us though, those are
                # not noise points, but outliers. So we will return them as well (unlike the clustered points,
                # their mean may be meaningless because they might be very far apart.
                noise_points = outlier_data_points[clustered == -1]
                if len(noise_points) > 0:
                    noise_points = noise_points.flatten().tolist()
                    cluster_means.extend(noise_points)
                # Return the cluster centers as the highlight meaning "outliers around these values"
                return True, (cluster_means, None)

            return True, (outlier_data_points, None)
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