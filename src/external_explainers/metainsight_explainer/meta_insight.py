from collections import defaultdict
from typing import List, Dict

import math

from external_explainers.metainsight_explainer.commoness_and_exceptions import categorize_exceptions, EXCEPTION_CATEGORY_COUNT
from external_explainers.metainsight_explainer.data_pattern import HomogenousDataPattern
from external_explainers.metainsight_explainer.data_pattern import BasicDataPattern

COMMONNESS_THRESHOLD = 0.5
BALANCE_PARAMETER = 1
ACTIONABILITY_REGULARIZER_PARAM = 0.1

class MetaInsight:
    """
    Represents a MetaInsight (HDP, commonness_set, exceptions).
    """

    def __init__(self, hdp: HomogenousDataPattern,
                 commonness_set: Dict[BasicDataPattern, List[BasicDataPattern]],
                 exceptions: Dict[str, List[BasicDataPattern]], score=0):
        """
        :param hdp: list of BasicDataPattern objects
        :param commonness_set: A dictionary mapping commonness patterns to lists of BasicDataPattern objects
        :param exceptions: A dictionary mapping exception categories to lists of BasicDataPattern objects
        """
        self.hdp = hdp
        self.commonness_set = commonness_set
        self.exceptions = exceptions
        self.score = score

    def __repr__(self):
        return f"MetaInsight(score={self.score:.4f}, #HDP={len(self.hdp)}, #Commonness={len(self.commonness_set)}, #Exceptions={len(self.exceptions)})"

    @staticmethod
    def create_meta_insight(hdp: HomogenousDataPattern) -> 'MetaInsight' | None:
        """
        Evaluates the HDP and creates a MetaInsight object.
        :param hdp: A HomogenousDataPattern object.
        :return: A MetaInsight object if possible, None otherwise.
        """
        if len(hdp) == 0:
            return None

        # Group patterns by similarity
        similarity_groups = defaultdict(list)
        for dp in hdp:
            found_group = False
            for key in similarity_groups:
                # Check similarity with the first element of an existing group
                if dp.sim(similarity_groups[key][0]):
                    similarity_groups[key].append(dp)
                    found_group = True
                    break
            if not found_group:
                # Create a new group with this pattern as the first element (key)
                similarity_groups[dp].append(dp)

        # Identify commonness(es) based on the threshold
        commonness_set = []
        exceptions = []
        total_patterns_in_hdp = len(hdp)

        # Need to iterate through the original HDP to ensure all patterns are considered
        # and assigned to either commonness or exceptions exactly once.
        processed_patterns = set()
        for dp in hdp:
            if dp in processed_patterns:
                continue

            is_commonness = False
            for key, group in similarity_groups.items():
                if dp in group:
                    # An equivalence class is a commonness if it contains more than COMMONNESS_THRESHOLD of the HDP
                    if len(group) / total_patterns_in_hdp > COMMONNESS_THRESHOLD:
                        commonness_set.append(group)
                        for pattern in group:
                            processed_patterns.add(pattern)
                        is_commonness = True
                    break  # Found the group for this pattern

            if not is_commonness:
                # If the pattern wasn't part of a commonness, add it to exceptions
                exceptions.append(dp)
                processed_patterns.add(dp)

        # A valid MetaInsight requires at least one commonness
        if not commonness_set:
            return None

        # Categorize exceptions (optional for basic MetaInsight object, but needed for scoring)
        categorized_exceptions = categorize_exceptions(commonness_set, exceptions)

        return MetaInsight(hdp, commonness_set, categorized_exceptions)

    def calculate_conciseness(self) -> float:
        """
        Calculates the conciseness score of a MetaInsight.
        Based on the entropy of category proportions.
        """
        n = len(self.hdp)
        if n == 0:
            return 0

        # Calculate entropy
        S = 0
        commonness_proportions = []
        for group, patterns in self.commonness_set.items():
            if len(patterns) > 0:
                proportion = len(patterns) / n
                S += proportion * math.log2(proportion)
                commonness_proportions.append(proportion)

        exception_proportions = []
        for category, patterns in self.exceptions.items():
            if len(patterns) > 0:
                proportion = len(patterns) / n
                S += BALANCE_PARAMETER * (proportion * math.log2(proportion))
                exception_proportions.append(proportion)

        # Convert to positive entropy
        S = -S

        # Compute S* (the upper bound of S)
        threshold = ((1 - COMMONNESS_THRESHOLD) * math.e) / (math.pow(COMMONNESS_THRESHOLD, 1 / BALANCE_PARAMETER))
        if EXCEPTION_CATEGORY_COUNT > threshold:
            S_star = -math.log2(COMMONNESS_THRESHOLD) + (BALANCE_PARAMETER * EXCEPTION_CATEGORY_COUNT
                                                        * math.pow(COMMONNESS_THRESHOLD, 1 / BALANCE_PARAMETER)
                                                        * math.log2(math.e))
        else:
            S_star = - COMMONNESS_THRESHOLD * math.log(COMMONNESS_THRESHOLD) - (
                BALANCE_PARAMETER * (1 - COMMONNESS_THRESHOLD) * math.log2((1 - COMMONNESS_THRESHOLD) / EXCEPTION_CATEGORY_COUNT)
            )



        indicator_value = 1 if len(exception_proportions) == 0 else 0
        conciseness = 1 - ((S + ACTIONABILITY_REGULARIZER_PARAM * indicator_value) / S_star)

        # Ensure conciseness is within a reasonable range, e.g., [0, 1]
        return conciseness