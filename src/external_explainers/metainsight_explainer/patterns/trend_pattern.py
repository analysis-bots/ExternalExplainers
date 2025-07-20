from typing import List, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from external_explainers.metainsight_explainer.patterns.pattern_base_classes import PatternBase
from external_explainers.metainsight_explainer.utils import generate_color_shades

class TrendPattern(PatternBase):
    __name__ = "Trend pattern"

    @staticmethod
    def _visualize_many(plt_ax, colors, line_styles, patterns, labels, sorted_indices):
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)][1]

            # Plot the trend line using numeric positions
            trend_label = f"{label}"
            x_range = np.arange(len(sorted_indices))
            plt_ax.plot(x_range, pattern.slope * x_range + pattern.intercept,
                        linestyle=line_style, color=color, linewidth=2, label=trend_label + " (trend line)")

    @staticmethod
    def visualize_many(plt_ax, patterns: List['TrendPattern'], labels: List[str],
                       gb_col: str,
                       commonness_threshold,
                       agg_func,
                       exception_patterns: List['TrendPattern'] = None,
                       exception_labels: List[str] = None,
                       max_labels: int = 8,
                       max_common_categories: int = 3,
                       plot_num: int | None = None, alpha_data: float =0.7, show_data: bool = True) -> None:
        # Define a color cycle for lines
        colors = plt.cm.tab10.colors

        # Define line styles for additional differentiation.
        # Taken from the matplotlib docs.
        line_styles = [
            ('densely dotted', (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('densely dashed', (0, (5, 1))),
            ('densely dashdotted', (0, (3, 1, 1, 1))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

        # Prepare patterns with consistent numeric positions
        all_patterns = patterns + (exception_patterns if exception_patterns else [])
        index_to_position, sorted_indices = PatternBase.prepare_patterns_for_visualization(all_patterns)

        TrendPattern._visualize_many(
            plt_ax=plt_ax,
            colors=colors,
            line_styles=line_styles,
            patterns=patterns,
            labels=labels,
            sorted_indices=sorted_indices
        )

        # Set x-ticks to show original index values
        if sorted_indices:
            PatternBase.handle_sorted_indices(plt_ax, sorted_indices)

        # Compute the mean value across the data as a whole, and visualize that line, if show_data is True
        if show_data:
            overall_mean_series = PatternBase.compute_mean_series(patterns, index_to_position)
            mean_x_positions = [index_to_position.get(idx) for idx in overall_mean_series.index if
                                idx in index_to_position]
            mean_values = [overall_mean_series.loc[idx] for idx in overall_mean_series.index if
                           idx in index_to_position]
            plt_ax.plot(mean_x_positions, mean_values, color='black', alpha=alpha_data, linewidth=5,
                        label='Mean Over All Data')

        # If exception patterns are provided, visualize them
        if exception_patterns:
            exception_colors = generate_color_shades('Reds', len(exception_patterns))
            TrendPattern._visualize_many(
                plt_ax=plt_ax,
                colors=exception_colors,
                line_styles=line_styles,
                patterns=exception_patterns,
                labels=exception_labels,
                sorted_indices=sorted_indices
            )


        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

        common_description, exceptions_description = patterns[0].get_title_description()

        title = PatternBase.create_title(
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
        plt_ax.set_title(title, fontsize=22)

        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # Add legend
        plt_ax.legend()

    def __init__(self, source_series: pd.Series, type: Literal['Increasing', 'Decreasing'],
                 slope: float, intercept: float = 0, value_name: str = None):
        """
        Initialize the Trend pattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param type: The type of the pattern.
        :param slope: The slope of the trend.
        """
        super().__init__(source_series, value_name)
        self.type = type
        self.slope = slope
        self.intercept = intercept
        self.hash = None

    def visualize(self, plt_ax, title: str = None) -> None:
        """
        Visualize the trend pattern.
        :param plt_ax:
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name])
        if title is not None:
            plt_ax.set_title(title)
        else:
            plt_ax.set_title(
                f"{self.type} trend in {self.value_name} with slope {self.slope:.2f} and intercept {self.intercept:.2f}")

    def __eq__(self, other) -> bool:
        """
        Check if two TrendPattern objects are equal.
        :param other: Another TrendPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if they have the same type
        (increasing / decreasing) (we trust that comparisons will be done on the same series).
        """
        if not isinstance(other, TrendPattern):
            return False
        # We do not compare the slope and intercept - we only care about the type of trend
        return self.type == other.type

    def __repr__(self) -> str:
        """
        String representation of the TrendPattern.
        :return: A string representation of the TrendPattern.
        """
        return f"TrendPattern(type={self.type})"

    def __str__(self) -> str:
        """
        String representation of the TrendPattern.
        :return: A string representation of the TrendPattern.
        """
        return f"TrendPattern(type={self.type})"

    def __hash__(self) -> int:
        """
        Hash representation of the TrendPattern.
        :return: A hash representation of the TrendPattern.
        """
        if self.hash is not None:
            return self.hash
        self.hash = hash(f"TrendPattern(type={self.type})")
        return self.hash

    def get_highlight_indexes(self) -> List[int | str] | None:
        return None

    def get_title_description(self) -> tuple[str, str]:
        return f"a(n) {self.type.lower()} trend", "a different trend"