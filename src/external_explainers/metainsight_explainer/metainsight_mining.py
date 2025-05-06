import itertools
from typing import List
import numpy as np

from external_explainers.metainsight_explainer.meta_insight import (MetaInsight,
                                                                    ACTIONABILITY_REGULARIZER_PARAM,
                                                                    BALANCE_PARAMETER,
                                                                    COMMONNESS_THRESHOLD)

MIN_SCORE = 0.01

class MetaInsightMiner:



    """
    This class is responsible for the actual process of mining MetaInsights.
    """
    def __init__(self, k=5, min_score=MIN_SCORE, min_commonness=COMMONNESS_THRESHOLD, balance_factor=BALANCE_PARAMETER,
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

    def rank_metainsights(self, metainsight_candidates: List[MetaInsight]):
        """
        Rank the MetaInsights based on their scores.

        :param metainsight_candidates: A list of MetaInsights to rank.
        :return: A list of the top k MetaInsights.
        """
        # Sort candidates by score initially (descending)
        sorted_candidates = sorted(metainsight_candidates, key=lambda mi: mi.score, reverse=True)

        selected_metainsights = []
        candidate_set = set(sorted_candidates)

        # Greedy selection of MetaInsights.
        # We compute the total use of the currently selected MetaInsights, then how much a candidate would add to that.
        # We take the candidate that adds the most to the total use, repeating until we have k MetaInsights or no candidates left.
        while len(selected_metainsights) < self.k and candidate_set:
            best_candidate = None
            max_gain = -np.inf

            for candidate in candidate_set:
                total_use_approx = sum(mi.score for mi in selected_metainsights) - \
                                      sum(mi1.compute_pairwise_overlap_score(mi2) for mi1, mi2 in itertools.combinations(metainsight_candidates, 2))

                total_use_with_candidate = total_use_approx + (candidate.score - sum(mi.compute_pairwise_overlap_score(candidate) for mi in selected_metainsights))

                gain = total_use_with_candidate - total_use_approx

                if gain > max_gain:
                    max_gain = gain
                    best_candidate = candidate

            if best_candidate:
                selected_metainsights.append(best_candidate)
                candidate_set.remove(best_candidate)
            else:
                # No candidate provides a positive gain, or candidate_set is empty
                break

        return selected_metainsights