import typing
from enum import Enum
from typing import List

import pandas as pd
import numpy as np
from diptest import diptest
from scipy.stats import zscore
from external_explainers.metainsight_explainer.patterns import UnimodalityPattern, TrendPattern, OutlierPattern, \
    PatternBase
import pymannkendall as mk
from singleton_decorator import singleton
from external_explainers.metainsight_explainer.cache import Cache


class PatternType(Enum):
    """
    An enumeration of the types of patterns.
    """
    NONE = 0
    OTHER = 1
    UNIMODALITY = 2
    TREND = 3
    OUTLIER = 4


@singleton
class PatternEvaluator:
    """
    A class to evaluate different patterns in a series.
    """

    def __init__(self):
        self.cache = Cache()
        self.OUTLIER_ZSCORE_THRESHOLD = 2.0  # Z-score threshold for outlier detection
        self.TREND_SLOPE_THRESHOLD = 0.01  # Minimum absolute slope for trend detection

    def _is_time_series(self, series: pd.Series) -> bool:
        """
        Checks if the series is a time series.
        We consider a series to be a time series if its index is either a datetime index or an increasing integer index.
        The second case is not always accurate, since an ordered series of numbers may not be a time series, but
        we also can not discard the possibility that it is a time series.
        :param series: The series to check.
        :return: True if the series is a time series, False otherwise.
        """
        if isinstance(series.index, pd.DatetimeIndex):
            return True
        elif np.issubdtype(series.index.dtype, np.number):
            # Check if the index is strictly increasing
            return np.all(np.diff(series.index) > 0)
        else:
            return False

    def unimodality(self, series: pd.Series) -> (bool, List[UnimodalityPattern] | None):
        """
        Evaluates if the series is unimodal using Hartigan's Dip test.
        If it is, finds the peak or valley.
        :param series: The series to evaluate.
        :return: Tuple (is_unimodal, UnimodalityPattern or None if not unimodal)
        """
        if isinstance(series, pd.Series):
            series = series.sort_index()
        else:
            return False, None
        vals = series.values
        if len(vals) < 4:
            return False, None
        # Perform Hartigan's Dip test
        dip_statistic, p_value = diptest(vals)
        is_unimodal = p_value > 0.05
        if not is_unimodal:
            return False, None
        # If there is unimodality, find the valley / peak
        max_value = series.max()
        min_value = series.min()
        # Check to make sure either the max or min happens only once, and is not at the start or end of the series
        peaks = series[series == max_value]
        valleys = series[series == min_value]
        if len(peaks) > 1 and len(valleys) > 1:
            return False, None
        max_value_index = peaks.index[0] if len(peaks) == 1 else None
        min_value_index = valleys.index[0] if len(valleys) == 1 else None
        # If both are at the edges, this is more likely a trend than a unimodal pattern
        if (max_value_index is not None and (
                max_value_index == series.index[0] or max_value_index == series.index[-1])) and \
                (min_value_index is not None and (
                        min_value_index == series.index[0] or min_value_index == series.index[-1])):
            return False, None
        to_return = []
        # If both a peak and a valley exists, we can return both. If none exists, we return None.
        if max_value_index:
            to_return.append(UnimodalityPattern(series, 'Peak', max_value_index, value_name=series.name))
        elif min_value_index:
            to_return.append(UnimodalityPattern(series, 'Valley', min_value_index, value_name=series.name))
        if len(to_return) == 0:
            return False, None
        return True, frozenset(to_return)

    def trend(self, series: pd.Series) -> (bool, TrendPattern | None):
        """
            Evaluates if a time series exhibits a significant trend (upward or downward).
            Uses the Mann-Kendall test to check for monotonic trends.

            :param series: The series to evaluate.
            :return: Tuple (trend_detected, a Trend pattern object or None. None if no trend is detected)
            """
        if len(series) < 2:
            return False, None

        # Check if the series is a time series
        if not self._is_time_series(series):
            return False, None

        # Use the Mann Kendall test to check for trend.
        mk_result = mk.original_test(series)
        p_val = mk_result.p
        # Reject or accept the null hypothesis
        if p_val > 0.05 or mk_result.trend == 'no trend':
            return False, None
        else:
            return True, TrendPattern(series, type=mk_result.trend,
                                      slope=mk_result.slope, intercept=mk_result.intercept, value_name=series.name)

    def outlier(self, series: pd.Series) -> (bool, OutlierPattern):
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
        outlier_indices = np.where(z_scores > self.OUTLIER_ZSCORE_THRESHOLD)[0]
        if len(outlier_indices) == 0:
            return False, None
        outlier_values = series.iloc[outlier_indices]
        outlier_indexes = series.index[outlier_indices]
        return True, OutlierPattern(series, outlier_indexes=outlier_indexes,
                                    outlier_values=outlier_values, value_name=series.name
                                    )

    def __call__(self, series: pd.Series, pattern_type: PatternType) -> (bool, frozenset[PatternBase] | None):
        """
        Calls the appropriate pattern evaluation method based on the pattern type.
        :param series: The series to evaluate.
        :param pattern_type: The type of the pattern to evaluate.
        :return: (is_valid, highlight)
        """
        series_hash = hash(tuple(series.values))
        cache_key = (series_hash, pattern_type)

        cache_result = self.cache.get_from_pattern_eval_cache(cache_key)
        if cache_result is not None:
            # If the result is already cached, return it
            return cache_result

        series = series[~series.isna()]  # Remove NaN values
        series = series.sort_index()  # Sort the series by index

        if pattern_type == PatternType.UNIMODALITY:
            result = self.unimodality(series)
        elif pattern_type == PatternType.TREND:
            result = self.trend(series)
        elif pattern_type == PatternType.OUTLIER:
            result = self.outlier(series)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
        is_valid = result[0] if isinstance(result, tuple) else False
        patterns = result[1] if isinstance(result, tuple) else None
        # If the returned patterns are not a frozenset, convert them to one
        if not isinstance(patterns, frozenset):
            if not isinstance(patterns, typing.Iterable):
                patterns = frozenset([patterns])
            else:
                patterns = frozenset(patterns)
        # Add the result to the cache
        self.cache.add_to_pattern_eval_cache(cache_key, (is_valid, patterns))
        return is_valid, patterns
