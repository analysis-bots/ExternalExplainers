from cmath import isnan
from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap

import math

from external_explainers.metainsight_explainer.data_pattern import HomogenousDataPattern
from external_explainers.metainsight_explainer.data_pattern import BasicDataPattern
from external_explainers.metainsight_explainer.pattern_evaluations import PatternType

COMMONNESS_THRESHOLD = 0.5
BALANCE_PARAMETER = 1
ACTIONABILITY_REGULARIZER_PARAM = 0.1
EXCEPTION_CATEGORY_COUNT = 3


class MetaInsight:
    """
    Represents a MetaInsight (HDP, commonness_set, exceptions).
    """

    def __init__(self, hdp: HomogenousDataPattern,
                 commonness_set: List[BasicDataPattern],
                 exceptions: Dict[str, List[BasicDataPattern]], score=0,
                 commonness_threshold: float = COMMONNESS_THRESHOLD,
                 balance_parameter: float = BALANCE_PARAMETER,
                 actionability_regularizer_param: float = ACTIONABILITY_REGULARIZER_PARAM,
                 source_name: str = None,
                 ):
        """
        :param hdp: list of BasicDataPattern objects
        :param commonness_set: A dictionary mapping commonness patterns to lists of BasicDataPattern objects
        :param exceptions: A dictionary mapping exception categories to lists of BasicDataPattern objects
        """
        self.hdp = hdp
        self.commonness_set: List[BasicDataPattern] = commonness_set
        self.exceptions = exceptions
        self.score = score
        self.commonness_threshold = commonness_threshold
        self.balance_parameter = balance_parameter
        self.actionability_regularizer_param = actionability_regularizer_param
        self.source_name = source_name if source_name else "df"
        self.hash = None

    def __repr__(self):
        return f"MetaInsight(score={self.score:.4f}, #HDP={len(self.hdp)}, #Commonness={len(self.commonness_set)}, #Exceptions={len(self.exceptions)})"

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        self.hash = 0
        for pattern in self.commonness_set:
            self.hash += pattern.__hash__()
        return self.hash


    def __eq__(self, other):
        """
        Compares two MetaInsight objects for equality.
        Two MetaInsight objects are considered equal if they have the same commonness sets.
        :param other:
        :return:
        """
        if not isinstance(other, MetaInsight):
            return False
        # If the commonness sets are not the same size, they are not equal
        if len(self.commonness_set) != len(other.commonness_set):
            return False
        all_equal = True
        # Do a two sided comparison of the commonness sets.
        for self_pattern in self.commonness_set:
            if self_pattern not in other.commonness_set:
                all_equal = False
                break
        for other_pattern in other.commonness_set:
            if other_pattern not in self.commonness_set:
                all_equal = False
                break

        return all_equal


    def __str__(self):
        """
        :return: A string representation of the MetaInsight, describing its commonness set and exceptions.
        """
        self_highlight = self.commonness_set[0].highlight
        common_description, exception_description = self_highlight.get_title_description()
        gb_col = self.commonness_set[0].data_scope.breakdown
        highlight_change_exceptions, exception_labels = self._get_highlight_change_exceptions()
        title = self_highlight.create_title(
            common_patterns=[pattern.highlight for pattern in self.commonness_set],
            common_patterns_labels=self._create_labels(self.commonness_set),
            gb_col=gb_col,
            agg_func=self.hdp[0].data_scope.measure[1],
            commonness_threshold=self.commonness_threshold,
            highlight_indexes=self_highlight.get_highlight_indexes(),
            common_pattern_description=common_description,
            exception_pattern_description=exception_description,
            exception_patterns=[pattern.highlight for pattern in highlight_change_exceptions] if highlight_change_exceptions else None,
            exception_patterns_labels=exception_labels if highlight_change_exceptions else None,
        )
        return title


    def _get_highlight_change_exceptions(self) -> tuple[List[BasicDataPattern], List[str]] | tuple[None, None]:
        """
        Get the highlight change exceptions for this MetaInsight.
        :return: A tuple containing the highlight change exceptions and their labels.
        """
        gb_col = self.commonness_set[0].data_scope.breakdown
        highlight_change_exceptions = [pattern for pattern in self.exceptions.get("Highlight-Change", [])
                                       if pattern.data_scope.breakdown == gb_col and
                                       pattern.data_scope.measure == self.commonness_set[0].data_scope.measure]
        if not highlight_change_exceptions:
            return None, None
        exception_labels = self._create_labels(highlight_change_exceptions)
        return highlight_change_exceptions, exception_labels


    def _write_exceptions_list_string(self, category: PatternType, patterns: List[BasicDataPattern], category_name: str) -> str:
        """
        Helper function to create a string representation of a list of exception patterns.
        :param category: The category of the exceptions.
        :param patterns: The list of BasicDataPattern objects in this category.
        :param category_name: The name of the category.
        :return: A string representation of the exceptions in this category.
        """
        if not patterns:
            return ""
        if category_name.lower() not in ["no pattern", "no-pattern", "none", "highlight-change", "highlight change"]:
            # If the category is "No Pattern" or "Highlight Change", we don't need to write anything
            exceptions = [pattern for pattern in patterns if pattern.pattern_type not in [PatternType.NONE, PatternType.OTHER]]
        else:
            exceptions = [pattern for pattern in patterns if pattern.pattern_type == category]
        subspaces = [pattern.data_scope.subspace for pattern in exceptions]
        subspace_dict = defaultdict(list)
        for subspace in subspaces:
            for key, val in subspace.items():
                subspace_dict[key].append(val)
        out_str = f"Exceptions in category '{category_name}' ({len(exceptions)}): ["
        for key, val in subspace_dict.items():
            out_str += f"{key} = {val}, "
        out_str = out_str[:-2] + "]\n"
        return out_str

    def get_exceptions_string(self):
        """
        A string representation of the list of exception categories.
        :return:
        """
        exceptions_string = ""
        for category, patterns in self.exceptions.items():
            if not patterns:
                continue
            # No-Pattern category: create an array of
            if category.lower() == "no-pattern" or category.lower() == "none":
                exceptions_string += self._write_exceptions_list_string(PatternType.NONE, patterns, "No Pattern")
            if category.lower() == "highlight-change" or category.lower() == "highlight change":
                # Doesn't matter which PatternType we use here, so long as it is not None or PatternType.OTHER.
                exceptions_string += self._write_exceptions_list_string(PatternType.UNIMODALITY, patterns, "Same pattern, different highlight")
            elif category.lower() == "type-change" or category.lower() == "type change":
                exceptions_string += self._write_exceptions_list_string(PatternType.OTHER, patterns, "Pattern type change")
        if not exceptions_string:
            exceptions_string = "All values belong to a commonness set, no exceptions found."
        return exceptions_string


    def to_str_full(self):
        """
        :return: A full string representation of the MetaInsight, including commonness sets and exceptions.
        """
        ret_str = self.__str__()
        if len(self.exceptions) > 0:
            ret_str += f"Exceptions to this pattern were found:\n"
        ret_str += self.get_exceptions_string()
        return ret_str


    @staticmethod
    def categorize_exceptions(commonness_set: List[BasicDataPattern], exceptions):
        """
        Categorizes exceptions based on differences from commonness highlights/types.
        Simplified categorization: Highlight-Change, Type-Change, No-Pattern (though No-Pattern
        should ideally not be in the exceptions list generated by generate_hdp).
        Returns a dictionary mapping category names to lists of exception patterns.
        """
        categorized = defaultdict(list)
        commonness_highlights = set()
        commonness_highlights.add(str(commonness_set[0]))  # Assume all in commonness have same highlight

        for exc_dp in exceptions:
            if exc_dp.pattern_type == PatternType.OTHER:
                categorized['Type-Change'].append(exc_dp)
            elif exc_dp.pattern_type == PatternType.NONE:
                categorized['No-Pattern'].append(exc_dp)
            elif str(exc_dp.highlight) not in commonness_highlights:
                categorized['Highlight-Change'].append(exc_dp)

        return categorized

    @staticmethod
    def create_meta_insight(hdp: HomogenousDataPattern, commonness_threshold=COMMONNESS_THRESHOLD) -> List['MetaInsight'] | None:
        """
        Evaluates the HDP and creates a MetaInsight object.
        :param hdp: A HomogenousDataPattern object.
        :param commonness_threshold: The threshold for commonness.
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
                    if len(group) / total_patterns_in_hdp > commonness_threshold:
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

        returned_metainsights = [
            MetaInsight(
                hdp=hdp,
                commonness_set=commonness,
                exceptions=MetaInsight.categorize_exceptions(commonness, exceptions),
                commonness_threshold=commonness_threshold,
            )
            for commonness in commonness_set
        ]

        return returned_metainsights

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
        if len(self.commonness_set) > 0:
            proportion = len(self.commonness_set) / n
            S += proportion * math.log2(proportion)
            commonness_proportions.append(proportion)

        exception_proportions = []
        for category, patterns in self.exceptions.items():
            if len(patterns) > 0:
                proportion = len(patterns) / n
                S += self.balance_parameter * (proportion * math.log2(proportion))
                exception_proportions.append(proportion)

        # Convert to positive entropy
        S = -S

        # Compute S* (the upper bound of S), using the formula from the paper
        threshold = ((1 - self.commonness_threshold) * math.e) / (
            math.pow(self.commonness_threshold, 1 / self.balance_parameter))
        if EXCEPTION_CATEGORY_COUNT > threshold:
            S_star = -math.log2(self.commonness_threshold) + (self.balance_parameter * EXCEPTION_CATEGORY_COUNT
                                                              * math.pow(self.commonness_threshold,
                                                                         1 / self.balance_parameter)
                                                              * math.log2(math.e))
        else:
            S_star = - self.commonness_threshold * math.log(self.commonness_threshold) - (
                    self.balance_parameter * (1 - self.commonness_threshold) * math.log2(
                (1 - self.commonness_threshold) / EXCEPTION_CATEGORY_COUNT)
            )

        indicator_value = 1 if len(exception_proportions) == 0 else 0
        conciseness = 1 - ((S + self.actionability_regularizer_param * indicator_value) / S_star)

        # Ensure conciseness is within a reasonable range, e.g., [0, 1]
        return conciseness

    def compute_score(self) -> float:
        """
        Computes the score of the MetaInsight.
        The score is the multiple of the conciseness of the MetaInsight and the impact score of the HDS
        making up the HDP.
        :param impact_measure: The impact measure to be used for the HDS.
        :return: The score of the MetaInsight.
        """
        conciseness = self.calculate_conciseness()
        # If the impact has already been computed, use it
        hds_score = self.hdp.impact if self.hdp.impact != 0 else self.hdp.compute_impact()
        self.score = conciseness * hds_score
        return self.score

    def compute_pairwise_overlap_ratio(self, other: 'MetaInsight') -> float:
        """
        Computes the pairwise overlap ratio between two MetaInsights, as the ratio between the
        size of the intersection and the size of the union of their HDPs.
        :param other: Another MetaInsight object to compare with.
        :return: The overlap ratio between the two MetaInsights.
        """
        if not isinstance(other, MetaInsight):
            raise ValueError("The other object must be an instance of MetaInsight.")
        hds_1 = set(self.hdp.data_scopes)
        hds_2 = set(other.hdp.data_scopes)

        overlap = len(hds_1.intersection(hds_2))
        total = len(hds_1.union(hds_2))
        # Avoid division by 0
        if total == 0:
            return 0.0
        return overlap / total

    def compute_pairwise_overlap_score(self, other: 'MetaInsight') -> float:
        """
        Computes the pairwise overlap score between two MetaInsights.
        This is computed as min(I_1.score, I_2.scor) * overlap_ratio(I_1, I_2)
        :param other: Another MetaInsight object to compare with.
        :return: The pairwise overlap score between the two MetaInsights.
        """
        if not isinstance(other, MetaInsight):
            raise ValueError("The other object must be an instance of MetaInsight.")
        overlap_ratio = self.compute_pairwise_overlap_ratio(other)
        if self.score == float('inf') or other.score == float('inf') or isnan(self.score) or isnan(other.score):
            # If either score is infinite or NaN, return 0 to avoid infinite overlap score
            return 0.0
        return min(self.score, other.score) * overlap_ratio


    def _create_labels(self, patterns: List[BasicDataPattern]) -> List[str]:
        """
        Create labels for the patterns in a commonness set.
        :param patterns: A list of BasicDataPattern objects.
        :return: A list of strings representing the labels for the patterns.
        """
        labels = []
        for pattern in patterns:
            subspace_str = ""
            for key, val in pattern.data_scope.subspace.items():
                if isinstance(val, str):
                    split = val.split("<=")
                    # If the value is a range (e.g. "0.1 <= x <= 0.5"), split it and format it by the range values.
                    # Otherwise, just use the value as is.
                    if len(split) > 1:
                        subspace_str += f"({split[0], split[2]})"
                    else:
                        subspace_str += f"{val}"
                else:
                    subspace_str += f"{val}"

            labels.append(f"{subspace_str}")
        return labels

    def visualize(self, plt_ax, plot_num: int | None = None) -> None:
        """
        Visualize the metainsight, showing commonness sets on the left and exceptions on the right.

        :param fig: Matplotlib figure to plot on. If None, a new figure is created.
        :param subplot_spec: GridSpec to plot on. If None, a new GridSpec is created.
        :param figsize: Size of the figure if a new one is created.
        :param additional_text: Optional additional text to display in the bottom-middle of the figure.
        """
        # Get the highlights for visualization
        highlights = [pattern.highlight for pattern in self.commonness_set]

        # Create labels based on subspace
        labels = self._create_labels(self.commonness_set)

        gb_col = self.commonness_set[0].data_scope.breakdown
        agg_func = self.commonness_set[0].data_scope.measure[1]

        # If there are highlight change exceptions of the same type, we add them to the visualization
        if len(self.exceptions) > 0 and "Highlight-Change" in self.exceptions:
            # Get the highlight change exceptions for this commonness set
            highlight_change_exceptions, exception_labels = self._get_highlight_change_exceptions()
            if highlight_change_exceptions:
                highlight_change_exceptions = [pattern.highlight for pattern in highlight_change_exceptions]
        else:
            highlight_change_exceptions = None
            exception_labels = None

        # Call the appropriate visualize_many function based on pattern type
        if highlights:
            if hasattr(highlights[0], "visualize_many"):
                highlights[0].visualize_many(plt_ax=plt_ax, patterns=highlights, labels=labels,
                                             gb_col=gb_col,
                                             agg_func=agg_func, commonness_threshold=self.commonness_threshold,
                                             exception_patterns=highlight_change_exceptions,
                                            exception_labels=exception_labels, plot_num=plot_num
                                             )

        return plt_ax

