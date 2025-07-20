import typing

import pandas as pd
from typing import Dict, List, Tuple

from external_explainers.metainsight_explainer.data_scope import DataScope, HomogenousDataScope
from external_explainers.metainsight_explainer.pattern_evaluations import PatternEvaluator, PatternType
from external_explainers.metainsight_explainer.patterns import PatternBase
from external_explainers.metainsight_explainer.cache import Cache


class BasicDataPattern:
    """
    A data pattern, as defined in the MetaInsight paper.
    Contains 3 elements: data scope, type (interpretation type) and highlight.
    """
    cache = Cache()

    def __init__(self, data_scope: DataScope, pattern_type: PatternType, highlight: PatternBase | None):
        """
        Initialize the BasicDataPattern with the provided data scope, type and highlight.

        :param data_scope: The data scope of the pattern. a DataScope object.
        :param pattern_type: str, e.g., 'Unimodality', 'Trend', 'Other Pattern', 'No Pattern'
        :param highlight: depends on type, e.g., ('April', 'Valley') for Unimodality
        """
        self.data_scope = data_scope
        self.pattern_type = pattern_type
        self.highlight = highlight
        self.hash = None

    def __eq__(self, other):
        if not isinstance(other, BasicDataPattern):
            return False
        return self.pattern_type == other.pattern_type and \
            self.highlight == other.highlight and \
            self.data_scope == other.data_scope

    def sim(self, other) -> bool:
        """
        Computes the similarity between two BasicDataPattern objects.
        They are similar if they have the same pattern type and highlight, as well as neither having
        a pattern type of NONE or OTHER.

        :param other: The other BasicDataPattern object to compare with.
        :return: True if similar, False otherwise.
        """
        if not isinstance(other, BasicDataPattern):
            return False
        # There is no REAL need to check that both don't have NONE or OTHER pattern types, since if one
        # has it but the other doesn't, the equality will be false anyway. If they both have it, then
        # the equality conditions will be true but the inequality conditions will be false.
        return self.pattern_type == other.pattern_type and self.highlight == other.highlight and \
            self.pattern_type != PatternType.NONE and self.pattern_type != PatternType.OTHER

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        self.hash = hash((hash(self.data_scope), self.pattern_type, self.highlight))
        return self.hash

    def __repr__(self):
        return f"BasicDataPattern(ds={self.data_scope}, type='{self.pattern_type}', highlight={self.highlight})"

    @staticmethod
    def evaluate_pattern(data_scope: DataScope, df: pd.DataFrame, pattern_type: PatternType) -> List['BasicDataPattern']:
        """
        Evaluates a specific pattern type for the data distribution of a data scope.
        :param data_scope: The data scope to evaluate.
        :param df: The DataFrame containing the data.
        :param pattern_type: The type of the pattern to evaluate.
        """
        # Apply subspace filters
        filtered_df = data_scope.apply_subspace()

        # Group by breakdown dimension and aggregate measure
        if any([dim for dim in data_scope.breakdown if dim not in filtered_df.columns]):
            # Cannot group by breakdown if it's not in the filtered data
            return [BasicDataPattern(data_scope, PatternType.NONE, None)]

        measure_col, agg_func = data_scope.measure
        if measure_col not in filtered_df.columns:
            # Cannot aggregate if measure column is not in the data
            return [BasicDataPattern(data_scope, PatternType.NONE, None)]

        try:
            # Perform the aggregation
            if agg_func != "std":
                aggregated_series = filtered_df.groupby(data_scope.breakdown)[measure_col].agg(agg_func)
            else:
                # For standard deviation, we need to use the std function directly
                aggregated_series = filtered_df.groupby(data_scope.breakdown)[measure_col].std(ddof=1)
        except Exception as e:
            print(f"Error during aggregation for {data_scope}: {e}")
            return [BasicDataPattern(data_scope, PatternType.NONE, None)]

        # Ensure series is sortable if breakdown is temporal
        if all([True for dim in data_scope.breakdown if df[dim].dtype.kind in 'iuMmfc']):
            # If the breakdown is temporal or at-least can be sorted, sort the series
            aggregated_series = aggregated_series.sort_index()

        # Evaluate the specific pattern type
        returned_patterns = []
        pattern_evaluator = PatternEvaluator()
        is_valid, highlight = pattern_evaluator(aggregated_series, pattern_type)
        if is_valid:
            # A returned highlight can contain multiple highlights, for example, if a peak and a valley are found
            # in the same series.
            for hl in highlight:
                returned_patterns.append(BasicDataPattern(data_scope, pattern_type, hl))
        else:
            # Check for other pattern types
            for other_type in PatternType:
                if other_type == PatternType.OTHER or other_type == PatternType.NONE:
                    continue
                if other_type != pattern_type:
                    other_is_valid, highlight = pattern_evaluator(aggregated_series, other_type)
                    if other_is_valid:
                        for hl in highlight:
                            returned_patterns.append(BasicDataPattern(data_scope, PatternType.OTHER, hl))

        if len(returned_patterns) == 0:
            # If no pattern is found, return a 'No Pattern' type
            return [BasicDataPattern(data_scope, PatternType.NONE, None)]

        return returned_patterns

    def create_hdp(self, pattern_type: PatternType,
                   hds: List[DataScope] = None, group_by_dims: List[List[str]] = None,
                   measures: List[Tuple[str,str]] = None, n_bins: int = 10,
                   extend_by_measure: bool = False, extend_by_breakdown: bool = False) -> 'HomogenousDataPattern':
        """
        Generates a Homogenous Data Pattern (HDP) either from a given HDS or from the current DataScope.

        :param pattern_type: The type of the pattern (e.g., 'Unimodality', 'Trend', etc.), provided as a PatternType enum.
        :param hds: A list of DataScopes to create the HDP from. If None, it will be created from the current DataScope.
        :param group_by_dims: The temporal dimensions to extend the breakdown with. Expected as a list of lists of strings.
        :param measures: The measures to extend the measure with. Expected to be a dict {measure_column: aggregate_function}. Only needed if hds is None.
        :param n_bins: The number of bins to use for numeric columns. Defaults to 10.
        :param extend_by_measure: Whether to extend the hds by measure. Defaults to False.
        :param extend_by_breakdown: Whether to extend the hds by breakdown. Defaults to False.
        :return: The HomogenousDataPattern object containing the evaluated patterns.
        """
        if hds is None or len(hds) == 0:
            hds = self.data_scope.create_hds(dims=group_by_dims, measures=measures,
                                             n_bins=n_bins, extend_by_measure=extend_by_measure,
                                             extend_by_breakdown=extend_by_breakdown)
        # All the data scopes in the HDS should have the same source_df, and it should be
        # the same as the source_df of the current DataScope (otherwise, this pattern should not be
        # the one producing the HDP with this HDS).
        source_df = self.data_scope.source_df

        # Create the HDP
        hdp = []
        for ds in hds:
            if ds != self.data_scope:
                # Check pattern cache first
                cache_key = (ds.__hash__(), pattern_type)
                cache_result = self.cache.get_from_pattern_cache(cache_key)
                if cache_result is not None:
                    dp = cache_result
                else:
                    # Evaluate the pattern if not in cache, and add to cache
                    dp = self.evaluate_pattern(ds, source_df, pattern_type)
                    self.cache.add_to_pattern_cache(cache_key, dp)

                # Some evaluation functions can return multiple patterns, so it is simpler to just
                # convert it to a list and then treat it as an iterable.
                if not isinstance(dp, typing.Iterable):
                    dp = [dp]

                # Add all patterns, including 'No Pattern', since it is important to know that we had a 'No Pattern'.
                for d in dp:
                    if dp is not None:
                        hdp.append(d)

        if self.pattern_type != PatternType.NONE:
            # Add the current pattern to the HDP
            hdp.append(self)
        hdp = HomogenousDataPattern(hdp)

        return hdp


class HomogenousDataPattern(HomogenousDataScope):
    """
    A homogenous data pattern.
    A list of data patterns induced by the same pattern type on a homogenous data scope.
    """

    def __init__(self, data_patterns: List[BasicDataPattern]):
        """
        Initialize the HomogenousDataPattern with the provided data patterns.

        :param data_patterns: A list of BasicDataPattern objects.
        """
        if not data_patterns:
            raise ValueError("data_patterns cannot be empty.")
        super(HomogenousDataPattern, self).__init__([dp.data_scope for dp in data_patterns])
        self.data_patterns = data_patterns

    def __iter__(self):
        """
        Allows iteration over the data patterns.
        """
        return iter(self.data_patterns)

    def __len__(self):
        """
        Returns the number of data patterns.
        """
        return len(self.data_patterns)

    def __repr__(self):
        return f"HomogenousDataPattern(#Patterns={len(self.data_patterns)})"

    def __getitem__(self, item):
        """
        Allows indexing into the data patterns.
        """
        return self.data_patterns[item]
