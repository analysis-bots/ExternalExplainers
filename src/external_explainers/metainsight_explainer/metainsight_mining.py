import itertools
from cmath import isnan
from typing import List, Tuple
import numpy as np
from queue import PriorityQueue

import pandas as pd
from matplotlib import pyplot as plt, gridspec

from external_explainers.metainsight_explainer.data_pattern import BasicDataPattern
from external_explainers.metainsight_explainer.meta_insight import (MetaInsight,
                                                                    ACTIONABILITY_REGULARIZER_PARAM,
                                                                    BALANCE_PARAMETER,
                                                                    COMMONNESS_THRESHOLD)
from external_explainers.metainsight_explainer.data_scope import DataScope
from external_explainers.metainsight_explainer.pattern_evaluations import PatternType
from external_explainers.metainsight_explainer.cache import Cache

MIN_IMPACT = 0.01


class MetaInsightMiner:
    """
    This class is responsible for the actual process of mining MetaInsights.
    The full process is described in the paper " MetaInsight: Automatic Discovery of Structured Knowledge for
    Exploratory Data Analysis" by Ma et al. (2021).
    """

    def __init__(self, k=5, min_score=MIN_IMPACT, min_commonness=COMMONNESS_THRESHOLD, balance_factor=BALANCE_PARAMETER,
                 actionability_regularizer=ACTIONABILITY_REGULARIZER_PARAM
                 ):
        """
        Initialize the MetaInsightMiner with the provided parameters.

        :param min_score: The minimum score for a MetaInsight to be considered.
        :param min_commonness: The minimum commonness for a MetaInsight to be considered.
        :param balance_factor: The balance factor for the MetaInsight.
        :param actionability_regularizer: The actionability regularizer for the MetaInsight.
        """
        self.k = k
        self.min_score = min_score
        self.min_commonness = min_commonness
        self.balance_factor = balance_factor
        self.actionability_regularizer = actionability_regularizer

    def _compute_variety_factor(self, metainsight: MetaInsight, included_pattern_types_count: dict) -> float:
        """
        Compute the variety factor for a given MetaInsight based on the pattern types
        already present in the selected set.

        :param metainsight: The MetaInsight object to compute the variety factor for.
        :param included_pattern_types_count: Dictionary tracking count of selected pattern types.
        :return: The variety factor between 0 and 1.
        """
        # Get pattern types in this metainsight
        candidate_pattern_type = metainsight.commonness_set[0].pattern_type

        # Check if this MetaInsight's pattern type is already included
        pattern_repetition = included_pattern_types_count.get(candidate_pattern_type, 0)
        if pattern_repetition == 0:
            return 1

        # Exponential decay: variety_factor decreases as pattern repetition increases
        # The 0.5 constant controls how quickly the penalty grows
        variety_factor = np.exp(-0.5 * pattern_repetition / self.k)

        return variety_factor


    def rank_metainsights(self, metainsight_candidates: List[MetaInsight]):
        """
        Rank the MetaInsights based on their scores.

        :param metainsight_candidates: A list of MetaInsights to rank.
        :return: A list of the top k MetaInsights.
        """

        selected_metainsights = []
        # Sort candidates by score initially (descending)
        candidate_set = sorted(list(set(metainsight_candidates)), key=lambda mi: mi.score, reverse=True)

        included_pattern_types_count = {
            pattern_type: 0
            for pattern_type in PatternType if pattern_type != PatternType.NONE and pattern_type != PatternType.OTHER
        }

        # Greedy selection of MetaInsights.
        # We compute the total use of the currently selected MetaInsights, then how much a candidate would add to that.
        # We take the candidate that adds the most to the total use, repeating until we have k MetaInsights or no candidates left.
        while len(selected_metainsights) < self.k and candidate_set:
            best_candidate = None
            max_gain = -np.inf

            total_use_approx = sum(mi.score for mi in selected_metainsights) - \
                               sum(mi1.compute_pairwise_overlap_score(mi2) for mi1, mi2 in
                                   itertools.combinations(selected_metainsights, 2))

            for candidate in candidate_set:
                total_use_with_candidate = total_use_approx + (candidate.score - sum(
                    mi.compute_pairwise_overlap_score(candidate) for mi in selected_metainsights))

                # Rare case where gain can't be computed due to NaN or infinite values
                if total_use_with_candidate == float('inf') or total_use_approx == float('inf') or isnan(total_use_with_candidate) or isnan(total_use_approx):
                    gain = 0
                else:
                    gain = total_use_with_candidate - total_use_approx
                # Added penalty for repeating the same pattern types
                variety_factor = self._compute_variety_factor(candidate, included_pattern_types_count)
                gain *= variety_factor

                if gain > max_gain:
                    max_gain = gain
                    best_candidate = candidate

            if best_candidate:
                selected_metainsights.append(best_candidate)
                candidate_set.remove(best_candidate)
                # Store a counter for the pattern types of the selected candidates
                candidate_pattern_type = best_candidate.commonness_set[0].pattern_type
                if candidate_pattern_type in included_pattern_types_count:
                    included_pattern_types_count[candidate_pattern_type] += 1
            else:
                # No candidate provides a positive gain, or candidate_set is empty
                break

        return selected_metainsights

    def mine_metainsights(self, source_df: pd.DataFrame,
                          filter_dimensions: List[str],
                          measures: List[Tuple[str,str]], n_bins: int = 10,
                          extend_by_measure: bool = False,
                          extend_by_breakdown: bool = False,
                          breakdown_dimensions: List[List[str]] = None,
                          ) -> List[MetaInsight]:
        """
        The main function to mine MetaInsights.
        Mines metainsights from the given data frame based on the provided dimensions, measures, and impact measure.
        :param source_df: The source DataFrame to mine MetaInsights from.
        :param breakdown_dimensions: The dimensions to consider for breakdown (groupby).
        :param filter_dimensions: The dimensions to consider for applying filters on.
        :param measures: The measures (aggregations) to consider for mining.
        :param n_bins: The number of bins to use for numeric columns.
        :param extend_by_measure: Whether to extend the data scope by measure. Settings this to true can cause strange results,
        because we will consider multiple aggregation functions on the same filter dimension.
        :param extend_by_breakdown: Whether to extend the data scope by breakdown. Settings this to true can cause strange results,
        because we will consider multiple different groupby dimensions on the same filter dimension, which can lead to
        having a metainsight on 2 disjoint sets of indexes.
        :return:
        """
        cache = Cache()
        hdp_queue = PriorityQueue()

        if breakdown_dimensions is None:
            breakdown_dimensions = filter_dimensions

        # Generate data scopes with one dimension as breakdown, all '*' subspace
        base_data_scopes = []
        for breakdown_dim in breakdown_dimensions:
            for measure_col, agg_func in measures:
                base_data_scopes.append(
                    DataScope(source_df, {}, breakdown_dim, (measure_col, agg_func)))

        # Generate data scopes with one filter in subspace and one breakdown
        for filter_dim in filter_dimensions:
            unique_values = source_df[filter_dim].dropna().unique()
            # If there are too many unique values, we bin them if it's a numeric column, or only choose the
            # top 10 most frequent values if it's a categorical column
            if len(unique_values) > n_bins:
                if source_df[filter_dim].dtype in ['int64', 'float64']:
                    # Bin the numeric column
                    bins = pd.cut(source_df[filter_dim], bins=n_bins, retbins=True)[1]
                    unique_values = [f"{bins[i]} <= {filter_dim} <= {bins[i + 1]}" for i in range(len(bins) - 1)]
                else:
                    # Choose the top 10 most frequent values
                    top_values = source_df[filter_dim].value_counts().nlargest(10).index.tolist()
                    unique_values = [v for v in unique_values if v in top_values]
            for value in unique_values:
                for breakdown_dim in breakdown_dimensions:
                    # Prevents the same breakdown dimension from being used as filter. This is because it
                    # is generally not very useful to groupby the same dimension as the filter dimension.
                    if breakdown_dim != filter_dim:
                        for measure_col, agg_func in measures:
                            base_data_scopes.append(
                                DataScope(source_df, {filter_dim: value}, breakdown_dim, (measure_col, agg_func)))

        # The source dataframe with a groupby on various dimensions and measures can be precomputed,
        # instead of computed each time we need it.
        numeric_columns = source_df.select_dtypes(include=[np.number]).columns.tolist()
        for col, agg_func in measures:
            groupby_key = (col, agg_func)
            cache_result = cache.get_from_groupby_cache(groupby_key)
            if cache_result is not None:
                # Handle 'std' aggregation specially
                if agg_func == 'std':
                    cache.add_to_groupby_cache(groupby_key, source_df.groupby(col)[numeric_columns].std(ddof=1))
                else:
                    cache.add_to_groupby_cache(groupby_key, source_df.groupby(col)[numeric_columns].agg(agg_func))


        for base_ds in base_data_scopes:
            # Evaluate basic patterns for the base data scope for selected types
            for pattern_type in PatternType:
                if pattern_type == PatternType.OTHER or pattern_type == PatternType.NONE:
                    continue
                base_dps = BasicDataPattern.evaluate_pattern(base_ds, source_df, pattern_type)

                for base_dp in base_dps:
                    if base_dp.pattern_type not in [PatternType.NONE, PatternType.OTHER]:
                        # If a valid basic pattern is found, extend the data scope to generate HDS
                        hdp = base_dp.create_hdp(group_by_dims=breakdown_dimensions, measures=measures,
                                                                pattern_type=pattern_type,
                                                                extend_by_measure=extend_by_measure, extend_by_breakdown=extend_by_breakdown)

                        # Pruning 1 - if the HDP is unlikely to form a commonness, discard it
                        if len(hdp) < len(hdp.data_scopes) * self.min_commonness:
                            continue

                        # Pruning 2: Discard HDS with extremely low impact
                        hds_impact = hdp.compute_impact()
                        if hds_impact < MIN_IMPACT:
                            continue

                        # Add HDS to a queue for evaluation
                        hdp_queue.put((hdp, pattern_type))

        metainsight_candidates = {}
        while not hdp_queue.empty():
            hdp, pattern_type = hdp_queue.get()

            # Evaluate HDP to find MetaInsight(s)
            metainsights = MetaInsight.create_meta_insight(hdp, commonness_threshold=self.min_commonness)

            if metainsights is not None:
                for metainsight in metainsights:
                    # Calculate and assign the score
                    metainsight.compute_score()
                    if metainsight in metainsight_candidates:
                        other_metainsight = metainsight_candidates[metainsight]
                        if metainsight.score > other_metainsight.score:
                            # If the new metainsight is better, replace the old one
                            metainsight_candidates[metainsight] = metainsight
                    else:
                        metainsight_candidates[metainsight] = metainsight

        return self.rank_metainsights(list(metainsight_candidates))


if __name__ == "__main__":
    # Create a sample Pandas DataFrame (similar to the paper's example)
    df = pd.read_csv("C:\\Users\\Yuval\\PycharmProjects\\pd-explain\\Examples\\Datasets\\adult.csv")
    df = df.sample(5000, random_state=42)  # Sample 5000 rows for testing
    print(df.columns)

    # Define dimensions, measures
    dimensions = ['education', 'occupation', 'marital-status']
    breakdown_dimensions = [['age'],
                            ['education-num'],
                            ['occupation'],
                            ['marital-status'],
                            ]
    measures = [
                ('capital-gain', 'mean'),
                ('capital-loss', 'mean'),
                ('hours-per-week', 'mean'),
                ('income', 'count'),
                ('education-num', 'mean'),
                ]

    # Run the mining process
    import time
    start_time = time.time()
    miner = MetaInsightMiner(k=4, min_score=0.01, min_commonness=0.5)
    top_metainsights = miner.mine_metainsights(
        source_df=df,
        filter_dimensions=dimensions,
        measures=measures,
        breakdown_dimensions=breakdown_dimensions,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    nrows = 2
    ncols = 2

    fig_len = 9 * ncols
    fig_height = 11 * nrows

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_len, fig_height), layout="constrained")

    for i, mi in enumerate(top_metainsights[:4]):
        row = i // ncols
        col = i % ncols
        mi.visualize(
            plt_ax=axs[row, col],
            plot_num=i + 1
        )

    # plt.tight_layout()
    plt.show()
