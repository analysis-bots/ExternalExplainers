from typing import List, Literal

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from external_explainers.metainsight_explainer.patterns.pattern_base_classes import PatternWithBarPlot
from external_explainers.metainsight_explainer.utils import generate_color_shades


class UnimodalityPattern(PatternWithBarPlot):
    __name__ = "Unimodality pattern"

    @staticmethod
    def _visualize_many(plt_ax, patterns: List['UnimodalityPattern'],
                        labels: List[str], colors, index_to_position,
                        total_series: int,
                        series_start_index: int,
                        alpha: float = 0.7, marker_size: int = 8) -> tuple[bool, bool]:
        """
        Internal method to visualize multiple unimodality patterns as a grouped bar plot.

        :param plt_ax: The matplotlib axes to plot on.
        :param patterns: List of UnimodalityPattern objects to visualize.
        :param labels: List of labels for each pattern.
        :param colors: List of colors to use for each pattern.
        :param index_to_position: Mapping from original indices to numeric positions for plotting.
        :param total_series: Total number of bar series in the grouped bar plot.
        :param series_start_index: The starting index for this batch of patterns (for color/position offset).
        :param alpha: Opacity for the bars.
        :param marker_size: Size of the highlight marker.
        :returns: A tuple containing two booleans indicating if any peak or valley was drawn (in that order).
        """
        # 1. Calculate the width for each individual bar
        # We use 0.8 to leave some padding between the groups on the x-axis.

        valley_drawn, peak_drawn = False, False

        offsets = PatternWithBarPlot.draw_bar(
            plt_ax = plt_ax,
            total_series = total_series,
            patterns = patterns,
            labels = labels,
            colors = colors,
            index_to_position = index_to_position,
            series_start_index = series_start_index,
            alpha = alpha
        )

        for i, (pattern, label, offset) in enumerate(zip(patterns, labels, offsets)):
            # 4. Apply the same offset to the highlight markers
            if pattern.highlight_index in pattern.source_series.index:
                highlight_pos = index_to_position[pattern.highlight_index] + offset

                if pattern.type.lower() == 'peak':
                    marker_symbol = 'o'
                    peak_drawn = True
                else:  # Assumes 'valley'
                    marker_symbol = 'v'
                    valley_drawn = True

                plt_ax.plot(highlight_pos, pattern.source_series.loc[pattern.highlight_index],
                            marker_symbol, color='black', markersize=marker_size, zorder=3)  # Use black for visibility

        return peak_drawn, valley_drawn

    @staticmethod
    def visualize_many(plt_ax, patterns: List['UnimodalityPattern'],
                       labels: List[str],
                       gb_col: str,
                       commonness_threshold,
                       agg_func,
                       exception_patterns: List['UnimodalityPattern'] = None,
                       exception_labels: List[str] = None, plot_num: int = None,
                       max_labels: int = 8,
                       max_common_categories: int = 3,
                       ) -> None:
        # Define a color cycle for lines
        colors = generate_color_shades(
            'Greens', len(patterns)
        )

        highlight_indexes = [pattern.highlight_index for pattern in patterns]
        if exception_patterns is not None:
            highlight_indexes += [pattern.highlight_index for pattern in exception_patterns]

        # Prepare patterns with consistent numeric positions
        index_to_position, sorted_indices, pattern_means, labels = PatternWithBarPlot.prepare_patterns(patterns, labels,
                                                                                                     highlight_indexes, num_to_keep=max_labels,
                                                                                                     exception_patterns=exception_patterns,
                                                                                                    max_common_categories=max_common_categories)

        if pattern_means is not None:
            patterns = [
                UnimodalityPattern(source_series=mean_pattern,
                                   type=patterns[0].type,
                                   highlight_index=patterns[0].highlight_index,
                                   value_name=patterns[0].value_name
                                   )
                for mean_pattern in pattern_means
            ]

        peak_drawn, valley_drawn = False, False

        total_series = len(patterns) + (len(exception_patterns) if exception_patterns else 0)

        # Plot each pattern
        common_peak_drawn, common_valley_drawn = UnimodalityPattern._visualize_many(plt_ax, patterns, labels, colors,
                                           index_to_position, alpha=0.8,
                                           total_series=total_series, series_start_index=0)

        exception_peak_drawn, exception_valley_drawn = False, False

        # If there are exception patterns, plot them too, and make sure they are highlighted
        if exception_patterns is not None:
            highlighting_colors = generate_color_shades(
                'Reds', len(exception_patterns), start=0.3, end=0.9
            )
            # Plot exception patterns with a different style
            exception_peak_drawn, exception_valley_drawn = UnimodalityPattern._visualize_many(
                plt_ax, exception_patterns, exception_labels, highlighting_colors,
                index_to_position, alpha=1.0, marker_size=16,
                series_start_index=len(patterns),
                total_series=total_series
            )

        peak_drawn = common_peak_drawn or exception_peak_drawn
        valley_drawn = common_valley_drawn or exception_valley_drawn

        # Set x-ticks to show original index values
        if sorted_indices:
            PatternWithBarPlot.handle_sorted_indices(plt_ax, sorted_indices)

        # Set labels and title
        plt_ax.set_xlabel(patterns[0].index_name if patterns else 'Index')
        plt_ax.set_ylabel(patterns[0].value_name if patterns else 'Value')

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
        plt_ax.set_title(title, fontsize=22)

        # Get the handles and labels from the bars first
        handles, labels = plt_ax.get_legend_handles_labels()

        # Conditionally add proxy artists for the markers
        if peak_drawn:
            # Create an invisible line with the desired marker and label
            peak_proxy = Line2D([0], [0], marker='o', color='w', label='Peak',
                                markerfacecolor='black', markersize=8)
            handles.append(peak_proxy)

        if valley_drawn:
            valley_proxy = Line2D([0], [0], marker='v', color='w', label='Valley',
                                  markerfacecolor='black', markersize=8)
            handles.append(valley_proxy)

        # Create the legend with the combined list of handles
        plt_ax.legend(handles=handles)

        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

    def __init__(self, source_series: pd.Series, type: Literal['Peak', 'Valley'], highlight_index,
                 value_name: str = None):
        """
        Initialize the UnimodalityPattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param type: The type of the pattern. Either 'Peak' or 'Valley' is expected.
        :param highlight_index: The index of the peak or valley.
        :param value_name: The name of the value to display.
        """
        super().__init__(source_series, value_name)
        self.type = type
        self.highlight_index = highlight_index
        self.index_name = source_series.index.name if source_series.index.name else 'Index'
        self.hash = None

    def visualize(self, plt_ax, title: str = None) -> None:
        """
        Visualize the unimodality pattern.
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name])
        if title is not None:
            plt_ax.set_title(title)
        else:
            plt_ax.set_title(f"{self.type} at {self.highlight_index} in {self.value_name}")

    def __eq__(self, other) -> bool:
        """
        Check if two UnimodalityPattern objects are equal.
        :param other: Another UnimodalityPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if they have the same type,
        the same highlight index.
        """
        if not isinstance(other, UnimodalityPattern):
            return False
        if not type(self.highlight_index) == type(other.highlight_index):
            return False
        return (self.type == other.type and
                self.highlight_index == other.highlight_index)

    def __repr__(self) -> str:
        """
        String representation of the UnimodalityPattern.
        :return: A string representation of the UnimodalityPattern.
        """
        return f"UnimodalityPattern(type={self.type}, highlight_index={self.highlight_index})"

    def __str__(self) -> str:
        """
        String representation of the UnimodalityPattern.
        :return: A string representation of the UnimodalityPattern.
        """
        return f"UnimodalityPattern(type={self.type}, highlight_index={self.highlight_index})"

    def __hash__(self) -> int:
        """
        Hash representation of the UnimodalityPattern.
        :return: A hash representation of the UnimodalityPattern.
        """
        if self.hash is not None:
            return self.hash
        self.hash = hash(f"UnimodalityPattern(type={self.type}, highlight_index={self.highlight_index})")
        return self.hash


    def get_highlight_indexes(self) -> List[int | str] | str | int:
        """
        Get the highlight indexes for the pattern.
        :return: A list containing the highlight index.
        """
        return self.highlight_index if self.highlight_index is not None else []


    def get_title_description(self) -> tuple[str, str]:
        return f"a uni-modal {self.type.lower()}", "different uni-modalities"