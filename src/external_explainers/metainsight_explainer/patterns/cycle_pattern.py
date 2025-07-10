from typing import List, Literal

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from external_explainers.metainsight_explainer.patterns.pattern_base_classes import PatternInterface
from external_explainers.metainsight_explainer.utils import generate_color_shades

class CyclePattern(PatternInterface):
    __name__ = "Cycle pattern"

    @staticmethod
    def visualize_many(plt_ax, patterns: List['CyclePattern'], labels: List[str],
                       gb_col: str,
                       commonness_threshold,
                       agg_func,
                       exception_patterns: List['UnimodalityPattern'] = None,
                       exception_labels: List[str] = None,
                       alpha_cycles: float = 0.3, line_alpha: float = 0.8) -> None:
        """
        Visualize multiple cycle patterns on a single plot with common cycles highlighted.

        :param plt_ax: Matplotlib axes to plot on
        :param patterns: List of CyclePattern objects
        :param labels: List of labels for each pattern
        :param title: Optional custom title for the plot
        :param alpha_cycles: Opacity for the highlighted cycle regions
        :param line_alpha: Opacity for the time series lines
        """
        import numpy as np

        # Define a color cycle for lines
        colors = plt.cm.tab10.colors

        # Prepare patterns with consistent numeric positions
        index_to_position, sorted_indices = PatternInterface.prepare_patterns_for_visualization(patterns)

        # Color for common cycles
        common_cycle_color = 'darkviolet'

        # Plot each dataset and collect legend handles
        legend_handles = []
        legend_labels = []

        # First, identify time ranges covered by cycles for each pattern
        all_cycle_data = []

        for pattern in patterns:
            if hasattr(pattern, 'cycles') and not pattern.cycles.empty:
                for _, cycle in pattern.cycles.iterrows():
                    # Map to numeric positions
                    t_start_pos = index_to_position.get(cycle['t_start'], None)
                    t_end_pos = index_to_position.get(cycle['t_end'], None)
                    if t_start_pos is not None and t_end_pos is not None:
                        all_cycle_data.append((t_start_pos, t_end_pos))

        # Find common cycle periods (using numeric positions)
        common_periods = []
        if len(patterns) > 1 and all_cycle_data:
            # Get all unique numeric positions from starts and ends
            all_positions = sorted(list(set([pos for start, end in all_cycle_data for pos in [start, end]])))

            # Create additional points between positions if needed
            if len(all_positions) > 1:
                position_points = np.linspace(min(all_positions), max(all_positions), 100)
            else:
                position_points = all_positions

            # For each position point, check if it falls within a cycle for each pattern
            overlap_counts = np.zeros(len(position_points))

            for pattern in patterns:
                if hasattr(pattern, 'cycles') and not pattern.cycles.empty:
                    pattern_mask = np.zeros(len(position_points), dtype=bool)
                    for _, cycle in pattern.cycles.iterrows():
                        t_start_pos = index_to_position.get(cycle['t_start'], None)
                        t_end_pos = index_to_position.get(cycle['t_end'], None)
                        if t_start_pos is not None and t_end_pos is not None:
                            pattern_mask = pattern_mask | (
                                    (position_points >= t_start_pos) & (position_points <= t_end_pos))
                    overlap_counts += pattern_mask

            # Find regions where all patterns have a cycle
            common_mask = overlap_counts == len(patterns)

            # Find contiguous regions of common cycles
            if np.any(common_mask):
                changes = np.diff(np.concatenate(([0], common_mask.astype(int), [0])))
                start_indices = np.where(changes == 1)[0]
                end_indices = np.where(changes == -1)[0] - 1

                for start_idx, end_idx in zip(start_indices, end_indices):
                    common_periods.append((position_points[start_idx], position_points[end_idx]))

        # Plot each pattern
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]

            # Map series to numeric positions for plotting
            x_positions = [index_to_position[idx] for idx in pattern.source_series.index]
            values = pattern.source_series.values

            # Plot the time series
            line, = plt_ax.plot(x_positions, values, color=color, alpha=line_alpha, linewidth=2, label=label)
            legend_handles.append(line)
            legend_labels.append(label)

            # Highlight each cycle with a semi-transparent fill
            if hasattr(pattern, 'cycles') and not pattern.cycles.empty:
                # Add individual cycle legend element
                cycle_patch = plt.Rectangle((0, 0), 1, 1, color=color, alpha=alpha_cycles)

                for _, cycle in pattern.cycles.iterrows():
                    t_start_pos = index_to_position.get(cycle['t_start'], None)
                    t_end_pos = index_to_position.get(cycle['t_end'], None)

                    if t_start_pos is None or t_end_pos is None:
                        continue

                    # Check if this cycle overlaps with common cycles
                    is_common = any(
                        start <= t_start_pos <= end and start <= t_end_pos <= end
                        for start, end in common_periods
                    )

                    # Highlight the cycle only if it is not in the common cycles
                    if not is_common:
                        # Highlight the cycle region
                        plt_ax.axvspan(t_start_pos, t_end_pos, color=color, alpha=alpha_cycles)

        # Highlight common cycles
        if common_periods:
            for start, end in common_periods:
                plt_ax.axvspan(start, end, color=common_cycle_color, alpha=alpha_cycles * 1.5, zorder=-1)

            # Add legend item for common cycles
            common_patch = plt.Rectangle((0, 0), 1, 1, color=common_cycle_color, alpha=alpha_cycles * 1.5)
            legend_handles.append(common_patch)
            legend_labels.append('Common cycles (all patterns)')

        # Set x-ticks to show original index values
        if sorted_indices:
            PatternInterface.handle_sorted_indices(plt_ax, sorted_indices)

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

        default_title = "Multiple Cycle Patterns"
        title = ""
        plt_ax.set_title(title if title is not None else default_title)

        # Add legend
        plt_ax.legend(legend_handles, legend_labels)

        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # Ensure bottom margin for x-labels
        plt_ax.figure.subplots_adjust(bottom=0.15)

    def __init__(self, source_series: pd.Series, cycles: pd.DataFrame, value_name: str = None):
        """
        Initialize the Cycle pattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param cycles: The cycles detected in the series.
        """
        super().__init__(source_series, value_name)
        # Cycles is a dataframe with the columns: t_start, t_end, t_minimum, doc, duration
        self.cycles = cycles
        self.hash = None
        self._cycle_tuples = frozenset((row['t_start'], row['t_end']) for _, row in cycles.iterrows())

    def visualize(self, plt_ax, title: str = None):
        """
        Visualize the cycle pattern.
        :param plt_ax:
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name], alpha_cycles=0.5, line_alpha=0.8)
        if title is not None:
            plt_ax.set_title(title)
        else:
            plt_ax.set_title(f"Cycles in {self.value_name} at {self._cycle_tuples}")

    def __eq__(self, other):
        """
        Check if two CyclePattern objects are equal.
        :param other:
        :return: True if they are equal, False otherwise. They are considered equal if the cycles of one are a
        subset of the other or vice versa.
        """
        if not isinstance(other, CyclePattern):
            return False

        # Use precomputed cycle tuples instead of computing them each time
        return self._cycle_tuples.issubset(other._cycle_tuples) or other._cycle_tuples.issubset(self._cycle_tuples)

    def __repr__(self) -> str:
        """
        String representation of the CyclePattern.
        :return: A string representation of the CyclePattern.
        """
        return f"CyclePattern(cycles={self.cycles})"

    def __str__(self) -> str:
        """
        String representation of the CyclePattern.
        :return: A string representation of the CyclePattern.
        """
        return f"CyclePattern(cycles={self.cycles})"

    def __hash__(self) -> int:
        """
        Hash representation of the CyclePattern.
        :return: A hash representation of the CyclePattern.
        """
        if self.hash is not None:
            return self.hash
        # Create a hashable representation of the key cycle properties
        if len(self.cycles) == 0:
            return hash("empty_cycle")
        # Use a tuple of tuples for cycle start/end times
        cycle_tuples = tuple((row['t_start'], row['t_end']) for _, row in self.cycles.iterrows())
        self.hash = hash(cycle_tuples)
        return self.hash
