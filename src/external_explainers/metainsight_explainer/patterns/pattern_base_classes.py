from abc import ABC, abstractmethod
from collections import defaultdict
import textwrap
import pandas as pd
import numpy as np
from typing import List, Any
from sklearn.cluster import KMeans

class PatternBase(ABC):
    """
    Abstract base class for defining patterns.
    """

    def __init__(self, source_series: pd.Series, value_name: str = None):
        """
        Initialize the pattern with the source series.
        :param source_series: The source series to evaluate.
        """
        self.source_series = source_series
        self.index_name = source_series.index.name if source_series.index.name else 'Index'
        self.value_name = value_name if value_name else 'Value'
        self.hash = None

    @abstractmethod
    def visualize(self, plt_ax, title: str = None) -> None:
        """
        Visualize the pattern.
        """
        # Note for all the implementations below: all of them just use the visualize_many method internally,
        # because that one handles all the complex cases already and can also visualize just one pattern.
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def __eq__(self, other) -> bool:
        """
        Check if two patterns are equal
        :param other: Another pattern of the same type
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the pattern.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the pattern.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def __hash__(self) -> int:
        """
        Hash representation of the pattern.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def prepare_patterns_for_visualization(patterns):
        """
        Prepare patterns for visualization by creating a consistent numeric position mapping.
        Returns a mapping of original indices to numeric positions for plotting.

        :param patterns: List of pattern objects with source_series attribute
        :return: Dictionary mapping original indices to positions and sorted unique indices
        """
        # Collect all unique indices from all patterns
        all_indices = set()
        for pattern in patterns:
            all_indices.update(pattern.source_series.index)

        # Sort indices in their natural order - this works for dates, numbers, etc.
        sorted_indices = sorted(list(all_indices))

        # Create mapping from original index to position (0, 1, 2, ...)
        index_to_position = {idx: pos for pos, idx in enumerate(sorted_indices)}

        return index_to_position, sorted_indices

    @staticmethod
    def handle_sorted_indices(plt_ax, sorted_indices):
        """
        Handle setting x-ticks and labels for the plot based on sorted indices.
        :param plt_ax: The matplotlib axes to set ticks on
        :param sorted_indices: The sorted indices to use for x-ticks
        """
        # For large datasets, show fewer tick labels
        step = max(1, len(sorted_indices) // 10)
        positions = list(range(0, len(sorted_indices), step))
        tick_labels = [str(sorted_indices[pos]) for pos in positions]

        plt_ax.set_xticks(positions)
        plt_ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=16)

    @staticmethod
    @abstractmethod
    def visualize_many(plt_ax, patterns: List['PatternBase'], labels: List[str],
                       agg_func: str, commonness_threshold: float,
                       gb_col: str,
                       exception_patterns: List['PatternBase'] = None,
                       exception_labels: List[str] = None, plot_num: int = None,
                       max_labels: int = 8,
                       max_common_categories: int = 3,
                       ) -> None:
        """
        Visualize many patterns of the same type on the same plot.
        :param plt_ax: The matplotlib axes to plot on
        :param patterns: The patterns to plot
        :param labels: The labels to display in the legend.
        :param agg_func: Name of the aggregation function used (e.g. 'mean', 'sum') when creating the series that lead to the pattern discovery.
        :param commonness_threshold: Threshold for commonness (e.g. 0.5 for 50%) used when creating the MetaInsights.
        :param gb_col: The column used for grouping the data when creating the series that lead to the pattern discovery.
        :param exception_patterns: Patterns that are of the same type as the common patterns, but are exceptions to
        those common patterns. Should be greatly highlighted in the plot if not None.
        :param exception_labels: Labels for the exception patterns, if exception_patterns is not None.
        :param plot_num: Number of the current plot. If provided, will be added to the title for clarity.
        :param max_labels: Maximum number of labels in the x-axis to show in the plot.
        :param max_common_categories: Maximum number of common categories to show in the plot.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def compute_mean_series(patterns: List['PatternBase'],
                            index_to_position: dict, names: list[str] | str | None = None) -> pd.Series:
        """
        Compute the mean series across multiple patterns.
        :param patterns: List of PatternInterface objects
        :return: A pandas Series representing the mean values across all patterns
        """
        # Create a dictionary to hold the mean values for each index
        mean_dict = {
            idx: [] for idx in index_to_position.keys()
        }
        for idx in index_to_position:
            for pattern in patterns:
                if idx in pattern.source_series.index:
                    mean_dict[idx].append(pattern.source_series.loc[idx])
        # If names are provided, use them as the series name
        if names is not None:
            if isinstance(names, list):
                names = ', '.join(names)
            elif isinstance(names, str):
                names = names
            else:
                names = None
        # If there is a final ',' at the end of the names, remove it
        if isinstance(names, str):
            names = names.strip()
        if names and names.endswith(','):
            names = names[:-1]
        # Compute the overall mean series
        overall_mean_series = pd.Series(
            {idx: np.mean(values) for idx, values in mean_dict.items() if values},
            name=names if names is not None else 'Overall Mean Data',
            index=index_to_position
        )

        overall_mean_series = overall_mean_series.dropna()
        return overall_mean_series

    @staticmethod
    def group_similar_together(patterns: List['PatternBase'], index_to_position, num_groups: int = 3) \
            -> tuple[List[List['PatternBase']], List[int]]:
        """
        Group similar patterns together using KMeans clustering.
        :param patterns: List of PatternInterface objects to group
        :param num_groups: Number of groups to form
        :return: A list of lists, where each sublist contains patterns in the same group
        """
        cluterer = KMeans(n_clusters=num_groups)
        # Use index_to_position, where the keys contain the entire index of the series, to create a 2D array,
        # where each row corresponds to a pattern and each column corresponds to a position in the series.
        # This will allow us to cluster patterns based on their source series values, filling in NaNs where necessary.
        clutering_array = np.zeros((len(patterns), len(index_to_position)))
        for i, pattern in enumerate(patterns):
            # Fill the array with the values from the source series, using the index_to_position mapping
            for idx, pos in index_to_position.items():
                if idx in pattern.source_series.index:
                    clutering_array[i, pos] = pattern.source_series.loc[idx]
                else:
                    clutering_array[
                        i, pos] = 0  # Use 0 for missing values, since both NaN and infinity cause issues with KMeans
        # Fit the KMeans model
        cluterer.fit(clutering_array)
        # Get cluster labels
        labels = cluterer.labels_
        # Group patterns by their cluster labels
        grouped_patterns = {
            i: [] for i in range(num_groups)
        }
        for pattern, label in zip(patterns, labels):
            grouped_patterns[label].append(pattern)
        # Return the grouped patterns as a list of lists
        return [grouped_patterns[i] for i in range(num_groups)], labels

    @staticmethod
    def labels_to_grouped_labels(labels: List[str], pattern_labels: List[int]) -> defaultdict:
        """
        Collect the labels for each group, such that each group of patterns has an array
        containing all of the labels for the patterns in that group.
        :param labels: The string labels for each pattern
        :param pattern_labels: The integer group labels for each pattern
        :return: A defaultdict mapping group labels to a string saying Mean ({labels}) for each group
        """
        grouped_labels = defaultdict(list)
        for i, val in enumerate(pattern_labels):
            grouped_labels[val].append(labels[i])
        output_labels = defaultdict(str)
        for key in grouped_labels.keys():
            output_labels[key] = f"Mean ({', '.join(grouped_labels[key])})" if len(grouped_labels[key]) > 1 else \
            grouped_labels[key][0]
        return output_labels

    @staticmethod
    def prepare_patterns(patterns: List['PatternBase'],
                         labels: List[str], highlight_indexes: list, num_to_keep = 8,
                         max_common_categories: int = 3,
                         exception_patterns: List['PatternBase'] = None) \
            -> tuple[dict, List[int], List[pd.Series] | None, List[str]]:
        """
        Prepares patterns for visualization by creating a consistent numeric position mapping,
        grouping similar patterns together if there are more than 3, and computing the mean series for each group.
        :param patterns: List of PatternInterface objects to prepare.
        :param labels: Labels for the patterns, used for visualization.
        :param highlight_indexes: The indexes where the common pattern or an exception occurs, or a single index if there is only one.
        :param num_to_keep: The maximum number of indices to keep in the result.
        :param max_common_categories: The maximum number of common categories to show in the plot. If there are more than this
        number, they will be grouped together and their mean series will be computed.
        :param exception_patterns: List of exception patterns, if any. If None, no exceptions are considered.
        :return:
        """
        all_patterns = patterns + (exception_patterns if exception_patterns is not None else [])
        index_to_position, sorted_indices = PatternBase.prepare_patterns_for_visualization(all_patterns)
        index_to_position = PatternBase.prune_index(
            index_to_position,
            highlight_indexes=highlight_indexes,
            num_to_keep=num_to_keep  # Keep only the most relevant indices
        )
        sorted_indices = list(index_to_position.keys())

        if len(patterns) > max_common_categories:
            pattern_groupings, pattern_labels = PatternBase.group_similar_together(patterns,
                                                                                   index_to_position,
                                                                                   num_groups=max_common_categories)
            # Create the new labels for the grouped patterns
            grouped_labels = PatternBase.labels_to_grouped_labels(labels, pattern_labels)
            # Compute the mean series for each group
            pattern_means = [PatternBase.compute_mean_series(
                group, index_to_position,
                names=grouped_labels[i])
                for i, group in enumerate(pattern_groupings)
            ]
            # Assign the labels this way to make sure their order matches the patterns
            labels = [
                grouped_labels[i] for i, _ in enumerate(pattern_means)
            ]

            return index_to_position, sorted_indices, pattern_means, labels

        else:
            return index_to_position, sorted_indices, None, labels

    @staticmethod
    # @abstractmethod
    def create_title(common_patterns: List['PatternBase'], common_patterns_labels: List[str],
                     agg_func: str, commonness_threshold: float,
                     gb_col: str,
                     common_pattern_description: str,
                     highlight_indexes: List[int | str] | int | str | None,
                     exception_patterns: List['PatternBase'] = None,
                     exception_patterns_labels: List[str] = None,
                     exception_pattern_description: str | None = None,
                     ) -> str:
        """
        Create a title for the plot based on the common patterns and their labels.
        :param common_patterns: List of common patterns
        :param common_patterns_labels: Labels for the common patterns
        :param agg_func: Name of the aggregation function used (e.g. 'mean', 'sum') when creating the series that lead to the pattern discovery.
        :param commonness_threshold: Threshold for commonness (e.g. 0.5 for 50%) used when creating the MetaInsights.
        :param gb_col: The column used for grouping the data when creating the series that lead to the pattern discovery.
        :param common_pattern_description: A string for the common pattern's description, e.g. "a unimodal peak".
        :param highlight_indexes: The indexes where the common pattern occurs, or a single index if there is only one.
         Leave as None to leave it out of the title (useful for trend patterns, where the highlight indexes are not relevant).
        :param exception_patterns: List of exception patterns, if any
        :param exception_patterns_labels: Labels for the exception patterns, if any
        :param exception_pattern_description: A string for the exception pattern's description, e.g. "different unimodaliies".
        :return: A string title for the plot
        """
        title = ""
        value_name = common_patterns[0].value_name
        # Create the title based on the common patterns
        title += f"At-least {commonness_threshold * 100:.1f}% of {agg_func}({value_name}) grouped by {gb_col} have {common_pattern_description}"
        if highlight_indexes:
            title += f" at {highlight_indexes}"
        # If there are exception patterns, add them to the title
        if exception_patterns is not None:
            title += f", with {exception_pattern_description} in "
            for exception_label in exception_patterns_labels:
                title += f"{exception_label},"
        # If there is a final ',' at the end of the title, remove it
        if title.endswith(','):
            title = title[:-1]
        # Wrap the title to avoid it being too long
        title = textwrap.fill(title, width=50, break_long_words=True, replace_whitespace=False)
        return title

    @staticmethod
    def prune_index(index_to_position: dict[int, Any],
                    highlight_indexes: List[int] | int,
                    num_to_keep: int = 10):
        """
        Prune the index to only keep the most relevant positions based on the highlight indexes.
        Keeps the highlight indexes and the closest positions to them, up to a maximum of num_to_keep.
        :param index_to_position: Dictionary mapping original indices to numeric positions
        :param highlight_indexes: Important indices to keep (can be a single index or a list)
        :param num_to_keep: Maximum number of indices to keep in the result
        :return: A pruned dictionary with only the most relevant indices
        """
        # Convert highlight_indexes to a list if it's a single index
        if not isinstance(highlight_indexes, list):
            highlight_indexes = [highlight_indexes]

        # Ensure highlight indexes are in the index_to_position dictionary
        valid_highlight_indexes = list(set([idx for idx in highlight_indexes if idx in index_to_position]))

        # If there are already enough highlight indexes, return them directly
        if len(valid_highlight_indexes) >= num_to_keep:
            return {idx: index_to_position[idx] for idx in valid_highlight_indexes[:num_to_keep]}

        # If there are no valid highlight indexes, return a subset of the original dictionary
        if not valid_highlight_indexes:
            if len(index_to_position) <= num_to_keep:
                return index_to_position.copy()

            # Take evenly spaced indices if we need to reduce
            all_indices = sorted(list(index_to_position.keys()))
            step = max(1, len(all_indices) // num_to_keep)
            selected_indices = all_indices[::step][:num_to_keep]
            return {idx: index_to_position[idx] for idx in selected_indices}

        # Get the positions of highlight indexes
        highlight_positions = [index_to_position[idx] for idx in valid_highlight_indexes]

        # Calculate distances from each index's position to the nearest highlight position
        distances = {}
        for idx, pos in index_to_position.items():
            if idx in valid_highlight_indexes:
                distances[idx] = 0  # Highlight indices have zero distance
            else:
                # Find the minimum distance to any highlight position
                min_distance = min(abs(pos - highlight_pos) for highlight_pos in highlight_positions)
                distances[idx] = min_distance

        # Sort indices by distance
        sorted_indices = sorted(distances.keys(), key=lambda idx: distances[idx])

        # Keep the num_to_keep closest indices
        indices_to_keep = sorted_indices[:num_to_keep]

        # If any of the highlight indexes are not in the indices to keep, add them
        for idx in valid_highlight_indexes:
            if idx not in indices_to_keep:
                indices_to_keep.append(idx)

        # Create the pruned dictionary
        return {idx: i for i, idx in enumerate(indices_to_keep)}


    @abstractmethod
    def get_highlight_indexes(self) -> List[int | str] | None | str | int:
        """
        Get the highlight indexes for the pattern.
        :return: A list of highlight indexes, which are the indices of the source series, or None if the highlight
        indexes are not applicable for this pattern type.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_title_description(self) -> tuple[str, str]:
        """
        Get a description of the pattern for use in the title.
        :return: A tuple containing the common pattern description and the exception pattern description.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    __name__ = "PatternInterface"


class PatternWithBarPlot(PatternBase, ABC):
    """
    Parent class for patterns that are visualized as bar plots.
    Contains the draw_bar method that handles the bar plotting logic.
    Inheriting classes should call this method, and follow it up with any additional plotting logic they need.
    """

    @staticmethod
    def draw_bar(plt_ax, total_series: int, patterns: List['PatternWithBarPlot'],
                 labels: List[str], colors: List, index_to_position: dict,
                 series_start_index: int, alpha: float) -> List[float]:
        """
        Draws a grouped bar plot for multiple patterns.
        :param plt_ax: The matplotlib axes to plot on.
        :param total_series: How many series are in the plot, including the patterns and exceptions.
        :param patterns: The patterns to plot, which should be instances of PatternWithBarPlot.
        :param labels: The labels for each pattern, used in the legend.
        :param colors: The colors to use for each pattern, should be a list of colors.
        :param index_to_position: Mapping from original indices to numeric positions for plotting.
        :param series_start_index: The starting index for this batch of patterns (for color/position offset).
        :param alpha: Opacity for the bars in the bar plot.
        :return: A list of the offsets applied to each bar in the group.
        """
        total_group_width = 0.8
        bar_width = total_group_width / total_series

        offsets = []

        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            # The 'global' index of the series determines its position in the group
            current_series_index = series_start_index + i

            # Use the global index for consistent coloring
            color = colors[current_series_index % len(colors)]

            # 2. Calculate the horizontal offset for the current bar from the center of the group
            # This formula places the bars side-by-side around the original x-tick.
            offset = (current_series_index - (total_series - 1) / 2) * bar_width
            offsets.append(offset)

            # Map the pattern's data to its corresponding base positions
            base_x_positions = [index_to_position[idx] for idx in pattern.source_series.index if
                                idx in index_to_position]

            # Drop any values that do not have their index in index_to_position
            values = [pattern.source_series.loc[idx] for idx in pattern.source_series.index if idx in index_to_position]

            # 3. Apply the offset to create the final bar positions
            final_x_positions = [pos + offset for pos in base_x_positions]

            # Add the bar plot using the new positions and width
            plt_ax.bar(final_x_positions, values, color=color, alpha=alpha, label=label,
                       width=bar_width, edgecolor='black', linewidth=0.5)

        return offsets