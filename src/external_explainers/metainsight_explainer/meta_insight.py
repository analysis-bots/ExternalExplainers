from cmath import isnan
from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap

import math

import numpy as np

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
                 commonness_set: List[List[BasicDataPattern]],
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
        self.commonness_set = commonness_set
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
        for commonness in self.commonness_set:
            for pattern in commonness:
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
        for self_commonness in self.commonness_set:
            for other_commonness in other.commonness_set:
                # Check if the commonness sets are equal
                if len(self_commonness) != len(other_commonness):
                    all_equal = False
                    break
                for pattern in self_commonness:
                    if pattern not in other_commonness:
                        all_equal = False
                        break

        return all_equal


    def __str__(self):
        """
        :return: A string representation of the MetaInsight, describing all of the commonnesses in it.
        """
        ret_str = ""
        for commonness in self.commonness_set:
            ret_str += self._create_commonness_set_title(commonness)
        return ret_str


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
    def categorize_exceptions(commonness_set, exceptions):
        """
        Categorizes exceptions based on differences from commonness highlights/types.
        Simplified categorization: Highlight-Change, Type-Change, No-Pattern (though No-Pattern
        should ideally not be in the exceptions list generated by generate_hdp).
        Returns a dictionary mapping category names to lists of exception patterns.
        """
        categorized = defaultdict(list)
        commonness_highlights = set()
        for commonness in commonness_set:
            if commonness:  # Ensure commonness is not empty
                commonness_highlights.add(str(commonness[0].highlight))  # Assume all in commonness have same highlight

        for exc_dp in exceptions:
            if exc_dp.pattern_type == PatternType.OTHER:
                categorized['Type-Change'].append(exc_dp)
            elif exc_dp.pattern_type == PatternType.NONE:
                # This case should ideally not happen if generate_hdp filters 'No Pattern'
                categorized['No-Pattern'].append(exc_dp)
            elif str(exc_dp.highlight) not in commonness_highlights:
                categorized['Highlight-Change'].append(exc_dp)

            # Keeping this commented out, since I couldn't figure out what to do with something in this catch-all category.
            # For now it will be ignored, but it could maybe be useful.
            # else:
            #      # Exception has a valid pattern type and highlight, but didn't meet commonness threshold
            #      # This could be another category or grouped with Highlight-Change
            #      categorized['Other-Exception'].append(exc_dp) # Add a catch-all category

        return categorized

    @staticmethod
    def create_meta_insight(hdp: HomogenousDataPattern, commonness_threshold=COMMONNESS_THRESHOLD) -> 'MetaInsight':
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

        # Categorize exceptions (optional for basic MetaInsight object, but needed for scoring)
        categorized_exceptions = MetaInsight.categorize_exceptions(commonness_set, exceptions)

        return MetaInsight(hdp, commonness_set, categorized_exceptions, commonness_threshold=commonness_threshold)

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
        for patterns in self.commonness_set:
            if len(patterns) > 0:
                proportion = len(patterns) / n
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

        # Compute S* (the upper bound of S)
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


    def _create_commonness_set_title(self, commonness_set: List[BasicDataPattern]) -> str:
        """
        Create a title for the commonness set based on the patterns it contains.
        :param commonness_set: A list of BasicDataPattern objects.
        :return: A string representing the title for the commonness set.
        """
        if not commonness_set:
            return "No Patterns"
        title = ""
        # Check the type of the first pattern in the set. All patterns in the set should be of the same type.
        pattern_type = commonness_set[0].pattern_type
        if pattern_type == PatternType.UNIMODALITY:
            title += "Common unimodality detected - "
            umimodality = commonness_set[0].highlight
            type = umimodality.type
            index = umimodality.highlight_index
            title += f"common {type} at index {index} "
        elif pattern_type == PatternType.TREND:
            trend = commonness_set[0].highlight
            trend_type = trend.type
            title += f"Common {trend_type} trend detected "
        elif pattern_type == PatternType.OUTLIER:
            title += "Common outliers detected "
            outliers = [pattern.highlight for pattern in commonness_set]
            common_outlier_indexes = {}
            # Create a counter for the outlier indexes
            for outlier in outliers:
                if outlier.outlier_indexes is not None:
                    for idx in outlier.outlier_indexes:
                        if idx in common_outlier_indexes:
                            common_outlier_indexes[idx] += 1
                        else:
                            common_outlier_indexes[idx] = 1
            # Sort the outlier indexes by their count
            common_outlier_indexes = sorted(common_outlier_indexes.items(), key=lambda x: x[1], reverse=True)
            # Take the top 5 most common outlier indexes
            num_outliers = len(common_outlier_indexes)
            common_outlier_indexes = list(dict(common_outlier_indexes).keys())
            # If there are more than 5, truncate the list and add "..."
            if num_outliers > 5:
                common_outlier_indexes.append("...")
            title += f"at indexes {' / '.join(map(str, common_outlier_indexes))}: "
        elif pattern_type == PatternType.CYCLE:
            title += "Common cycles detected "
        # Find the common subspace of the patterns in the set
        # First, get the data scope of all of the patterns in the set
        data_scopes = [pattern.data_scope for pattern in commonness_set]
        subspaces = [datascope.subspace for datascope in data_scopes]
        # Now, find the common subspace they share.
        shared_subspace = set(subspaces[0].keys())
        for subspace in subspaces[1:]:
            shared_subspace.intersection_update(subspace.keys())
        title += f"for over {self.commonness_threshold * 100}% of values of {', '.join(shared_subspace)}, "
        breakdowns = set([str(datascope.breakdown) for datascope in data_scopes])
        measures = set([datascope.measure for datascope in data_scopes])
        measures_str = []
        for measure in measures:
            if isinstance(measure, tuple):
                measures_str.append(f"{{{measure[0]}: {measure[1]}}}")
            else:
                measures_str.append(measure)
        title += f"when grouping by {' or '.join(breakdowns)} and aggregating by {' or '.join(measures_str)}"
        title = textwrap.wrap(title, 70)
        title = "\n".join(title)
        return title

    def visualize_commonesses_individually(self, fig=None, subplot_spec=None, figsize=(15, 10)) -> None:
        """
        Visualize only the commonness sets of the metainsight, with each set in its own column.
        Within each column, patterns are arranged in a grid with at most 3 patterns per column.
        This was the initial visualization method, but it was too cluttered and not very useful, so it was renamed and
        replaced with the more compact and informative visualize method.

        :param fig: Optional figure to plot on (or create a new one if None)
        :param subplot_spec: Optional subplot specification to plot within
        :param figsize: Figure size if creating a new figure
        :return: The figure with visualization
        """
        # Create figure if not provided
        if fig is None:
            fig = plt.figure(figsize=figsize)

        # Only proceed if there are commonness sets
        if not self.commonness_set:
            return fig

        # Create the main grid with one column per commonness set
        num_commonness_sets = len(self.commonness_set)

        if subplot_spec is not None:
            # Use the provided subplot area
            outer_grid = gridspec.GridSpecFromSubplotSpec(1, num_commonness_sets,
                                                          subplot_spec=subplot_spec,
                                                          wspace=0.6, hspace=0.4)
        else:
            # Use the entire figure
            outer_grid = gridspec.GridSpec(1, num_commonness_sets, figure=fig, wspace=0.6, hspace=0.4)

        # For each commonness set
        for i, patterns in enumerate(self.commonness_set):
            # Calculate how many sub-columns needed for this set
            num_patterns = len(patterns)
            num_cols = math.ceil(num_patterns / 3)  # At most 3 patterns per column
            max_patterns_per_col = min(3, math.ceil(num_patterns / num_cols))

            # Create a sub-grid for this commonness set's title and patterns
            set_grid = gridspec.GridSpecFromSubplotSpec(
                max_patterns_per_col + 1,  # Title row + pattern rows
                num_cols,
                subplot_spec=outer_grid[i],
                height_ratios=[0.2] + [1] * max_patterns_per_col,  # Title row smaller
                hspace=1.5,  # Increased spacing between rows
                wspace=0.5,  # Increased spacing between columns
            )

            # Add the set title spanning all columns in the first row
            title_ax = fig.add_subplot(set_grid[0, :])
            set_title = self._create_commonness_set_title(patterns)
            title_ax.text(0.5, 0.5, set_title,
                          ha='center', va='center',
                          fontsize=12, fontweight='bold')
            title_ax.axis('off')  # Hide axis for the title

            # Plot each pattern
            j = 0
            for pattern in patterns:
                # Visualize the pattern
                if hasattr(pattern, 'highlight') and pattern.highlight is not None:
                    # Calculate which column and row this pattern should be in
                    col = j // max_patterns_per_col
                    row = (j % max_patterns_per_col) + 1  # +1 to skip title row
                    # Create subplot for this pattern
                    ax = fig.add_subplot(set_grid[row, col])

                    pattern.highlight.visualize(ax)

                    # Rotate x-axis tick labels
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

                    # Instead of setting title, add text box for query below the plot
                    query_text = pattern.data_scope.create_query_string(df_name=self.source_name)
                    query_text = textwrap.fill(query_text, width=40)

                    # Add text box with query string instead of title
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                    ax.text(0.5, 1.5, query_text, transform=ax.transAxes, fontsize=9,
                            ha='center', va='top', bbox=props)

                    j += 1

        return fig


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

    def visualize(self, fig=None, subplot_spec=None, figsize=(15, 10), additional_text: str = None) -> None:
        """
        Visualize the metainsight, showing commonness sets on the left and exceptions on the right.

        :param fig: Matplotlib figure to plot on. If None, a new figure is created.
        :param subplot_spec: GridSpec to plot on. If None, a new GridSpec is created.
        :param figsize: Size of the figure if a new one is created.
        :param additional_text: Optional additional text to display in the bottom-middle of the figure.
        """
        # Create a new figure if not provided
        # n_cols = 2 if self.exceptions and len(self.exceptions) > 0 else 1
        # Above line makes it so the plot of the commonness sets takes up the entire figure if there are no exceptions.
        # However, this can potentially make for some confusion, so I elected to always use 2 columns.
        n_cols = 2
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if subplot_spec is None:
            outer_grid = gridspec.GridSpec(1, n_cols, width_ratios=[1] * n_cols, figure=fig, wspace=0.2)
        else:
            outer_grid = gridspec.GridSpecFromSubplotSpec(1, n_cols, width_ratios=[1] * n_cols,
                                                          subplot_spec=subplot_spec, wspace=0.2)

        # Wrap the existing 1x2 layout in a 2-row local GridSpec
        if additional_text:
            wrapper_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=subplot_spec, height_ratios=[10, 1], hspace=0.8
            )
        else:
            wrapper_gs = gridspec.GridSpecFromSubplotSpec(
                1, 1, subplot_spec=subplot_spec
            )
        top_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=wrapper_gs[0], wspace=0.2
        )

        # Set up the left side for commonness sets
        left_grid = gridspec.GridSpecFromSubplotSpec(1, len(self.commonness_set),
                                                     subplot_spec=top_gs[0, 0], wspace=0.3)

        # Plot each commonness set in its own column
        for i, commonness_set in enumerate(self.commonness_set):
            if not commonness_set:  # Skip empty sets, though this should not happen
                continue

            # Create a subplot for this commonness set
            ax = fig.add_subplot(left_grid[0, i])

            # Add light orange background to commonness sets
            # ax.set_facecolor((1.0, 0.9, 0.8, 0.2))  # Light orange with alpha

            # Get the highlights for visualization
            highlights = [pattern.highlight for pattern in commonness_set]

            # Create labels based on subspace
            labels = self._create_labels(commonness_set)

            gb_col = commonness_set[0].data_scope.breakdown
            agg_func = commonness_set[0].data_scope.measure[1]

            # If there are highlight change exceptions of the same type, we add them to the visualization
            if len(self.exceptions) > 0 and "Highlight-Change" in self.exceptions:
                # Get the highlight change exceptions for this commonness set
                highlight_change_exceptions = [pattern for pattern in self.exceptions["Highlight-Change"]
                                               if pattern.data_scope.breakdown == gb_col and
                                               pattern.data_scope.measure == commonness_set[0].data_scope.measure]
                if highlight_change_exceptions:
                    exception_labels = self._create_labels(highlight_change_exceptions)
                    highlight_change_exceptions = [pattern.highlight for pattern in highlight_change_exceptions]
                else:
                    exception_labels = None
                    highlight_change_exceptions = None
            else:
                highlight_change_exceptions = None
                exception_labels = None

            # Call the appropriate visualize_many function based on pattern type
            if highlights:
                if hasattr(highlights[0], "visualize_many"):
                    highlights[0].visualize_many(plt_ax=ax, patterns=highlights, labels=labels,
                                                 gb_col=gb_col,
                                                 agg_func=agg_func, commonness_threshold=self.commonness_threshold,
                                                 exception_patterns=highlight_change_exceptions,
                                                exception_labels=exception_labels
                                                 )

        # Handle exceptions area if there are any
        if self.exceptions and n_cols > 1:
            none_patterns_exist = self.exceptions.get("No-Pattern", None) is not None
            # Set up the right side for exceptions with one row per exception type
            # If there are no exceptions, we create a grid with equal height ratios for each exception type.
            # Else, we create a grid where the last row is smaller if there are None exceptions.
            if not none_patterns_exist:
                right_grid = gridspec.GridSpecFromSubplotSpec(len(self.exceptions), 1,
                                                              subplot_spec=top_gs[0, 1],
                                                              hspace=1.2)  # Add more vertical space
            else:
                # If there are None exceptions, place them at the bottom with very little space, since it just text
                height_ratios = [10] * (len(self.exceptions) - 1) + [1] if len(self.exceptions) > 1 else [1]
                right_grid = gridspec.GridSpecFromSubplotSpec(len(self.exceptions), 1,
                                                              subplot_spec=outer_grid[0, 1],
                                                              height_ratios=height_ratios,
                                                              hspace=1.4)  # Add more vertical space
                # Get the None patterns and "summarize" them in a dictionary
                exception_patterns = self.exceptions.get("No-Pattern", [])
                non_exceptions = [pattern for pattern in exception_patterns if pattern.pattern_type == PatternType.NONE]
                non_exceptions_subspaces = [pattern.data_scope.subspace for pattern in non_exceptions]
                non_exceptions_dict = defaultdict(list)
                for subspace in non_exceptions_subspaces:
                    for key, val in subspace.items():
                        non_exceptions_dict[key].append(val)
                # Create a title for the None patterns
                title = f"No patterns detected ({len(non_exceptions)})"
                title = textwrap.fill(title, width=40)
                # Create text saying all the values for which no patterns were detected
                no_patterns_text = ""
                for key, val in non_exceptions_dict.items():
                    no_patterns_text += f"{key} = {val}\n"
                no_patterns_text = textwrap.fill(no_patterns_text, width=60)
                # Create a subplot for the None patterns
                ax = fig.add_subplot(right_grid[len(self.exceptions) - 1, 0])
                # Add title and text
                if len(self.exceptions) == 1:
                    title_y = None
                    text_y = 0.9
                else:
                    title_y = -0.3
                    text_y = -1.1
                text_x = 0.5
                ax.set_title(title, y=title_y, fontsize=18, fontweight='bold')
                ax.text(text_x, text_y, no_patterns_text,
                        ha='center', va='center',
                        fontsize=18)
                ax.axis('off')  # Hide axis for the title

            # Process each exception category
            i = 0
            for category, exception_patterns in self.exceptions.items():
                if not exception_patterns:  # Skip empty categories
                    continue

                # For "None" category, already handled it above
                if category.lower() == "none" or category.lower() == "no-pattern":
                    continue

                # # For "type change" or other categories, create a nested grid
                # elif category.lower() == "type-change" or category.lower() == "type change":
                #     # Make sure there are highlights to visualize
                #     highlights = [pattern.highlight for pattern in exception_patterns]
                #     if all(highlight is None for highlight in highlights):
                #         continue
                #
                #     # Create a nested grid for this row with more space
                #     type_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                #                                                  subplot_spec=right_grid[i, 0],
                #                                                  height_ratios=[1, 15], hspace=0.6, wspace=0.3)
                #
                #     # Add title for the category in the first row
                #     title_ax = fig.add_subplot(type_grid[0, 0])
                #     title_ax.axis('off')
                #     title_ax.set_facecolor((0.8, 0.9, 1.0, 0.2))
                #     title_ax.text(0.5, 0,
                #                   s=f"Different patterns types detected ({len(exception_patterns)})",
                #                   horizontalalignment='center',
                #                   verticalalignment='center',
                #                   fontsize=16,
                #                   fontweight='bold'
                #     )
                #
                #     # Create subplots for each pattern in the second row
                #     num_patterns = len(exception_patterns)
                #     # At most 2 patterns per row
                #     n_cols = 2 if num_patterns >= 2 else 1
                #     n_rows = math.ceil(num_patterns / n_cols)
                #     pattern_grid = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                #                                                     subplot_spec=type_grid[1, 0],
                #                                                     wspace=0.4, hspace=0.6)  # More horizontal space
                #
                #
                #     for j, pattern in enumerate(exception_patterns):
                #         col_index = j % n_cols
                #         row_index = j // n_cols
                #         ax = fig.add_subplot(pattern_grid[row_index, col_index])
                #         # ax.set_facecolor((0.8, 0.9, 1.0, 0.2))  # Light blue with alpha
                #
                #         # Format labels for title
                #         subspace_str = ", ".join([f"{key}={val}" for key, val in pattern.data_scope.subspace.items()])
                #
                #         title = f"{pattern.highlight.__name__} when {subspace_str}"
                #         title = "\n".join(textwrap.wrap(title, 30))  # Wrap title to prevent overflow
                #
                #         # Visualize the individual pattern with internal legend
                #         if pattern.highlight:
                #             pattern.highlight.visualize(ax, title=title)

                i += 1

        # If there is additional text, add it to the bottom middle of the grid
        if additional_text:
            text_ax = fig.add_subplot(wrapper_gs[1])
            text_ax.axis('off')
            text_ax.text(
                0.5, 0.5, additional_text,
                ha='center', va='center', fontsize=18
            )

        # Allow more space for the figure elements
        plt.subplots_adjust(bottom=0.15, top=0.9)  # Adjust bottom and top margins

        return fig

