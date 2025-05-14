import itertools
from typing import List, Tuple
import numpy as np
from queue import PriorityQueue

import pandas as pd
from matplotlib import pyplot as plt, gridspec

from external_explainers.metainsight_explainer.data_pattern import BasicDataPattern, HomogenousDataPattern
from external_explainers.metainsight_explainer.meta_insight import (MetaInsight,
                                                                    ACTIONABILITY_REGULARIZER_PARAM,
                                                                    BALANCE_PARAMETER,
                                                                    COMMONNESS_THRESHOLD)
from external_explainers.metainsight_explainer.data_scope import DataScope
from external_explainers.metainsight_explainer.pattern_evaluations import PatternType

MIN_IMPACT = 0.01


class MetaInsightMiner:
    """
    This class is responsible for the actual process of mining MetaInsights.
    The full process is described in the paper " MetaInsight: Automatic Discovery of Structured Knowledge for
    Exploratory Data Analysis" by Ma et al. (2021).
    """

    def __init__(self, k=5, min_score=MIN_IMPACT, min_commonness=COMMONNESS_THRESHOLD, balance_factor=BALANCE_PARAMETER,
                 actionability_regularizer=ACTIONABILITY_REGULARIZER_PARAM):
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
        candidate_pattern_types = [commonness[0].pattern_type for commonness in metainsight.commonness_set]

        if not candidate_pattern_types:
            return 0.0

        # Calculate how many of this metainsight's pattern types are already included
        pattern_repetition = [included_pattern_types_count.get(pt, 0) for pt in candidate_pattern_types]
        if any(pt == 0 for pt in pattern_repetition):
            return 1
        pattern_repetition = sum(pattern_repetition)

        # Normalize by the number of pattern types in this metainsight
        avg_repetition = pattern_repetition / len(candidate_pattern_types)

        # Exponential decay: variety_factor decreases as pattern repetition increases
        # The 0.5 constant controls how quickly the penalty grows
        variety_factor = np.exp(-0.5 * avg_repetition)

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
                candidate_pattern_types = [commonness[0].pattern_type for commonness in best_candidate.commonness_set]
                for pattern_type in candidate_pattern_types:
                    if pattern_type in included_pattern_types_count:
                        included_pattern_types_count[pattern_type] += 1
            else:
                # No candidate provides a positive gain, or candidate_set is empty
                break

        return selected_metainsights

    def mine_metainsights(self, source_df: pd.DataFrame,
                          dimensions: List[str],
                          measures: List[Tuple[str,str]]) -> List[MetaInsight]:
        """
        The main function to mine MetaInsights.
        Mines metainsights from the given data frame based on the provided dimensions, measures, and impact measure.
        :param source_df: The source DataFrame to mine MetaInsights from.
        :param dimensions: The dimensions to consider for mining.
        :param measures: The measures to consider for mining.
        :return:
        """
        metainsight_candidates = set()
        datascope_cache = {}
        pattern_cache = {}
        hdp_queue = PriorityQueue()

        # Generate data scopes with one dimension as breakdown, all '*' subspace
        base_data_scopes = []
        for breakdown_dim in dimensions:
            for measure_col, agg_func in measures:
                base_data_scopes.append(
                    DataScope(source_df, {}, breakdown_dim, (measure_col, agg_func)))

        # Generate data scopes with one filter in subspace and one breakdown
        for filter_dim in dimensions:
            unique_values = source_df[filter_dim].dropna().unique()
            # If there are too many unique values, we bin them if it's a numeric column, or only choose the
            # top 10 most frequent values if it's a categorical column
            if len(unique_values) > 10:
                if source_df[filter_dim].dtype in ['int64', 'float64']:
                    # Bin the numeric column
                    bins = pd.cut(source_df[filter_dim], bins=10, retbins=True)[1]
                    unique_values = [f"{bins[i]} <= {filter_dim} <= {bins[i + 1]}" for i in range(len(bins) - 1)]
                else:
                    # Choose the top 10 most frequent values
                    top_values = source_df[filter_dim].value_counts().nlargest(10).index.tolist()
                    unique_values = [v for v in unique_values if v in top_values]
            for value in unique_values:
                for breakdown_dim in dimensions:
                    # Prevents the same breakdown dimension from being used as filter. This is because it
                    # is generally not very useful to groupby the same dimension as the filter dimension.
                    if breakdown_dim != filter_dim:
                        for measure_col, agg_func in measures:
                            base_data_scopes.append(
                                DataScope(source_df, {filter_dim: value}, breakdown_dim, (measure_col, agg_func)))


        for base_ds in base_data_scopes:
            # Evaluate basic patterns for the base data scope for selected types
            for pattern_type in PatternType:
                if pattern_type == PatternType.OTHER or pattern_type == PatternType.NONE:
                    continue
                base_dp = BasicDataPattern.evaluate_pattern(base_ds, source_df, pattern_type)

                if base_dp.pattern_type not in [PatternType.NONE, PatternType.OTHER]:
                    # If a valid basic pattern is found, extend the data scope to generate HDS
                    hdp, pattern_cache = base_dp.create_hdp(temporal_dimensions=dimensions, measures=measures,
                                                            pattern_type=pattern_type, pattern_cache=pattern_cache)

                    # Pruning 1 - if the HDP is unlikely to form a commonness, discard it
                    if len(hdp) < len(hdp.data_scopes) * self.min_commonness:
                        continue

                    # Pruning 2: Discard HDS with extremely low impact
                    hds_impact = hdp.compute_impact(datascope_cache)
                    if hds_impact < MIN_IMPACT:
                        continue

                    # Add HDS to a queue for evaluation
                    hdp_queue.put((hdp, pattern_type))

        processed_hdp_count = 0
        while not hdp_queue.empty():
            hdp, pattern_type = hdp_queue.get()
            processed_hdp_count += 1

            # Evaluate HDP to find MetaInsight
            metainsight = MetaInsight.create_meta_insight(hdp, commonness_threshold=self.min_commonness)

            if metainsight:
                # Calculate and assign the score
                metainsight.compute_score(datascope_cache)
                metainsight_candidates.add(metainsight)

        return self.rank_metainsights(list(metainsight_candidates))


if __name__ == "__main__":
    # Create a sample Pandas DataFrame (similar to the paper's example)
    df = pd.read_csv("C:\\Users\\Yuval\\PycharmProjects\\pd-explain\\Examples\\Datasets\\adult.csv")
    df = df.sample(5000, random_state=42)  # Sample 5000 rows for testing

    # Define dimensions, measures
    dimensions = ['marital-status', 'workclass', 'education-num']
    measures = [('capital-gain', 'mean'), ('capital-loss', 'mean'),
                ('hours-per-week', 'mean')]

    # Run the mining process
    import time
    start_time = time.time()
    miner = MetaInsightMiner(k=4, min_score=0.01, min_commonness=0.5)
    top_metainsights = miner.mine_metainsights(
        df,
        dimensions,
        measures,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    fig = plt.figure(figsize=(30, 25))
    main_grid = gridspec.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.3)

    for i, mi in enumerate(top_metainsights[:4]):
        row, col = i // 2, i % 2
        mi.visualize_commonesses(fig=fig, subplot_spec=main_grid[row, col])

    # plt.tight_layout()
    plt.show()
