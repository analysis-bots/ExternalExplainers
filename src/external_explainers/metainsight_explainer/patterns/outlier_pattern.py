from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from external_explainers.metainsight_explainer.patterns.pattern_base_classes import PatternWithBarPlot
from external_explainers.metainsight_explainer.utils import generate_color_shades

class OutlierPattern(PatternWithBarPlot):
    __name__ = "Outlier pattern"

    @staticmethod
    def _visualize_many(plt_ax, patterns: List['OutlierPattern'],
                        labels: List[str], colors, index_to_position,
                        total_series: int,
                        series_start_index: int,
                        alpha: float = 0.7, marker_size: int = 8):
        """
        Internal method to visualize multiple outlier patterns as a grouped bar plot.

        :param plt_ax: The matplotlib axes to plot on.
        :param patterns: List of UnimodalityPattern objects to visualize.
        :param labels: List of labels for each pattern.
        :param colors: List of colors to use for each pattern.
        :param index_to_position: Mapping from original indices to numeric positions for plotting.
        :param total_series: Total number of bar series in the grouped bar plot.
        :param series_start_index: The starting index for this batch of patterns (for color/position offset).
        :param alpha: Opacity for the bars.
        :param marker_size: Size of the highlight marker.
        :returns: A tuple containing two booleans indicating if any valley or peak was drawn (in that order).
        """

        offsets = PatternWithBarPlot.draw_bar(
            plt_ax=plt_ax,
            total_series=total_series,
            patterns=patterns,
            labels=labels,
            colors=colors,
            index_to_position=index_to_position,
            series_start_index=series_start_index,
            alpha=alpha
        )

        # Draw highlight markers for outliers
        for i, (pattern, label, offset) in enumerate(zip(patterns, labels, offsets)):
            if pattern.outlier_indexes is not None and len(pattern.outlier_indexes) > 0:
                outlier_positions = [index_to_position[idx] + offset for idx in pattern.outlier_indexes]
                outlier_values = pattern.source_series.loc[pattern.outlier_indexes].values

                plt_ax.plot(
                    outlier_positions,
                    outlier_values,
                    'X',
                    color='black',
                    markersize=marker_size,
                    alpha=alpha,
                    zorder=3
                )

    @staticmethod
    def visualize_many(plt_ax, patterns: List['OutlierPattern'], labels: List[str],
                       gb_col: str,
                       commonness_threshold,
                       agg_func,
                       exception_patterns: List['OutlierPattern'] = None,
                       exception_labels: List[str] = None,
                       max_labels: int = 8,
                       max_common_categories: int = 3,
                       plot_num: int = None) -> None:
        regular_colors = generate_color_shades('Greens', len(patterns))


        highlight_indexes = [pattern.outlier_indexes for pattern in patterns]
        highlight_indexes = [idx for sublist in highlight_indexes for idx in sublist]
        if exception_patterns:
            exception_highlight_indexes = [pattern.outlier_indexes for pattern in exception_patterns]
            exception_highlight_indexes = [idx for sublist in exception_highlight_indexes for idx in sublist]
            highlight_indexes += exception_highlight_indexes
        # Prepare patterns with consistent numeric positions, and prune if necessary
        index_to_position, sorted_indices, pattern_means, labels = PatternWithBarPlot.prepare_patterns(patterns, labels,
                                                                                                       highlight_indexes,
                                                                                                       num_to_keep=max_labels,
                                                                                                       max_common_categories=max_common_categories,
                                                                                                       exception_patterns=exception_patterns)
        # Renew the patterns with the new pattern means
        if pattern_means is not None:
            patterns = [
                OutlierPattern(
                    source_series=mean_pattern,
                    outlier_indexes=patterns[0].outlier_indexes,
                    outlier_values= patterns[0].outlier_values,
                    value_name= patterns[0].value_name
                ) for mean_pattern in pattern_means
            ]

        total_series = len(patterns) + len(exception_patterns) if exception_patterns else len(patterns)

        # Visualize regular patterns
        OutlierPattern._visualize_many(
            plt_ax=plt_ax,
            patterns=patterns,
            labels=labels,
            colors=regular_colors,
            index_to_position=index_to_position,
            total_series=total_series,
            series_start_index=0,
            alpha=0.7,
            marker_size=12
        )

        # Visualize exception patterns if provided
        if exception_patterns:
            exception_colors = generate_color_shades('Reds', len(exception_patterns))
            OutlierPattern._visualize_many(
                plt_ax=plt_ax,
                patterns=exception_patterns,
                labels=exception_labels,
                colors=exception_colors,
                index_to_position=index_to_position,
                total_series=total_series,
                series_start_index=len(patterns),
                alpha=1,
                marker_size=16
            )

        # Set x-ticks to show original index values
        if sorted_indices:
            PatternWithBarPlot.handle_sorted_indices(plt_ax, sorted_indices)

        # Setup the rest of the plot
        custom_lines = [Line2D([0], [0], marker='X', color='black',
                               markerfacecolor='black', markersize=10, linestyle='')]
        custom_labels = ['Outliers (marked with X)']

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')


        common_description, exceptions_description = patterns[0].get_title_description()

        title = PatternWithBarPlot.create_title(
            common_patterns=patterns,
            common_patterns_labels=labels,
            agg_func=agg_func,
            commonness_threshold=commonness_threshold,
            gb_col=gb_col,
            highlight_indexes=patterns[0].get_highlight_indexes(),
            common_pattern_description=common_description,
            exception_patterns=exception_patterns,
            exception_patterns_labels=exception_labels,
            exception_pattern_description=exceptions_description
        )
        if plot_num is not None:
            title = f"[{plot_num}] {title}"
        plt_ax.set_title(title if title is not None else "Multiple Outlier Patterns", fontsize=22)

        # Setup legend
        handles, labels_current = plt_ax.get_legend_handles_labels()
        all_handles = handles + custom_lines
        all_labels = labels_current + custom_labels
        plt_ax.legend(all_handles, all_labels)

        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

    def __init__(self, source_series: pd.Series, outlier_indexes: pd.Index, outlier_values: pd.Series,
                 value_name: str = None):
        """
        Initialize the Outlier pattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param outlier_indexes: The indexes of the outliers.
        :param outlier_values: The values of the outliers.
        """
        super().__init__(source_series, value_name)
        self.outlier_indexes = outlier_indexes
        self.outlier_values = outlier_values
        self.hash = None

    def visualize(self, plt_ax, title: str = None) -> None:
        """
        Visualize the outlier pattern.
        :param plt_ax:
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name])
        if title is not None:
            plt_ax.set_title(title)
        else:
            plt_ax.set_title(f"Outliers in {self.value_name} at {self.outlier_indexes.tolist()}")

    def __eq__(self, other):
        """
        Check if two OutlierPattern objects are equal.
        :param other: Another OutlierPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if the index set of one is a subset
        of the other or vice versa.
        """
        if not isinstance(other, OutlierPattern):
            return False
        # If one index is a multi-index and the other is not, for example, they cannot be equal
        if not type(self.outlier_indexes) == type(other.outlier_indexes):
            return False
        return self.outlier_indexes.isin(other.outlier_indexes).all() or \
            other.outlier_indexes.isin(self.outlier_indexes).all()

    def __repr__(self) -> str:
        """
        String representation of the OutlierPattern.
        :return: A string representation of the OutlierPattern.
        """
        return f"OutlierPattern(outlier_indexes={self.outlier_indexes})"

    def __str__(self) -> str:
        """
        String representation of the OutlierPattern.
        :return: A string representation of the OutlierPattern.
        """
        return f"OutlierPattern(outlier_indexes={self.outlier_indexes})"

    def __hash__(self) -> int:
        """
        Hash representation of the OutlierPattern.
        :return: A hash representation of the OutlierPattern.
        """
        if self.hash is not None:
            return self.hash
        self.hash = hash(f"OutlierPattern(outlier_indexes={self.outlier_indexes})")
        return self.hash

    def get_highlight_indexes(self) -> List[int | str]:
        """
        Get the indexes of the outliers for highlighting.
        :return: A list of indexes of the outliers.
        """
        return self.outlier_indexes.tolist() if self.outlier_indexes is not None else []


    def get_title_description(self) -> tuple[str, str]:
        return "outliers", "different outliers"