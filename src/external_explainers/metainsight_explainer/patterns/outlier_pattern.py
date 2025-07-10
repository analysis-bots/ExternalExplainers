from typing import List, Literal

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from external_explainers.metainsight_explainer.patterns.pattern_base_classes import PatternWithBarPlot
from external_explainers.metainsight_explainer.utils import generate_color_shades

class OutlierPattern(PatternWithBarPlot):
    __name__ = "Outlier pattern"

    @staticmethod
    def visualize_many(plt_ax, patterns: List['OutlierPattern'], labels: List[str],
                       gb_col: str,
                       commonness_threshold,
                       agg_func,
                       exception_patterns: List['UnimodalityPattern'] = None,
                       exception_labels: List[str] = None,
                       show_regular: bool = True, alpha_regular: float = 0.5, alpha_outliers: float = 0.9) -> None:
        """
        Visualize multiple outlier patterns on a single plot.
        """
        colors = plt.cm.tab10.colors
        regular_marker = 'o'
        outlier_marker = 'X'

        # Prepare patterns with consistent numeric positions
        index_to_position, sorted_indices = PatternWithBarPlot.prepare_patterns_for_visualization(patterns)

        # Plot each pattern
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]

            # Plot regular data points
            if show_regular:
                # Get positions and values for plotting
                positions = [index_to_position[idx] for idx in pattern.source_series.index]
                values = pattern.source_series.values

                plt_ax.scatter(
                    positions,
                    values,
                    color=color,
                    alpha=alpha_regular,
                    marker=regular_marker,
                    s=30,
                    label=label
                )
            else:
                plt_ax.scatter([], [], color=color, marker=regular_marker, s=30, label=label)

            # Plot outliers
            if pattern.outlier_indexes is not None and len(pattern.outlier_indexes) > 0:
                # Map outliers to positions
                outlier_positions = []
                outlier_values = []

                for idx in pattern.outlier_indexes:
                    if idx in pattern.source_series.index:
                        outlier_positions.append(index_to_position[idx])
                        outlier_values.append(pattern.source_series.loc[idx])

                plt_ax.scatter(
                    outlier_positions,
                    outlier_values,
                    color=color,
                    alpha=alpha_outliers,
                    marker=outlier_marker,
                    s=100,
                    edgecolors='black',
                    linewidth=1.5
                )

        # Set x-ticks to show original index values
        if sorted_indices:
            PatternWithBarPlot.handle_sorted_indices(plt_ax, sorted_indices)

        # Setup the rest of the plot
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], marker=outlier_marker, color='black',
                               markerfacecolor='black', markersize=10, linestyle='')]
        custom_labels = ['Outliers (marked with X)']

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

        title = ""
        plt_ax.set_title(title if title is not None else "Multiple Outlier Patterns")

        # Setup legend
        handles, labels_current = plt_ax.get_legend_handles_labels()
        all_handles = handles + custom_lines
        all_labels = labels_current + custom_labels
        plt_ax.legend(all_handles, all_labels)

        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # Ensure bottom margin for x-labels
        plt_ax.figure.subplots_adjust(bottom=0.15)

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