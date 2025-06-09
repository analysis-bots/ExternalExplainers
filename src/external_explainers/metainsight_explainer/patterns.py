from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, List


class PatternInterface(ABC):
    """
    Abstract base class for defining patterns.
    """

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
    def visualize_many(plt_ax, patterns: List['PatternInterface'], labels:List[str], title: str = None) -> None:
        """
        Visualize many patterns of the same type on the same plot.
        :param plt_ax: The matplotlib axes to plot on
        :param patterns: The patterns to plot
        :param labels: The labels to display in the legend.
        :param title: The title of the plot
        """
        raise NotImplementedError("Subclasses must implement this method.")

    __name__ = "PatternInterface"


class UnimodalityPattern(PatternInterface):

    __name__ = "Unimodality pattern"

    @staticmethod
    def visualize_many(plt_ax, patterns: List['UnimodalityPattern'], labels: List[str], title: str = None) -> None:
        """
        Visualize multiple unimodality patterns on a single plot.

        :param plt_ax: Matplotlib axes to plot on
        :param patterns: List of UnimodalityPattern objects
        :param labels: List of labels for each pattern (e.g. data scope descriptions)
        """
        # Define a color cycle for lines
        colors = plt.cm.tab10.colors

        # Prepare patterns with consistent numeric positions
        index_to_position, sorted_indices = PatternInterface.prepare_patterns_for_visualization(patterns)

        # Plot each pattern
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]

            # Map series to numeric positions for plotting
            x_positions = [index_to_position[idx] for idx in pattern.source_series.index]
            values = pattern.source_series.values

            # Plot the series with a unique color
            plt_ax.plot(x_positions, values, color=color, alpha=0.7, label=label)

            # Highlight the peak or valley with a marker
            if pattern.type.lower() == 'peak' and pattern.highlight_index in pattern.source_series.index:
                highlight_pos = index_to_position[pattern.highlight_index]
                plt_ax.plot(highlight_pos, pattern.source_series.loc[pattern.highlight_index],
                            'o', color=color, markersize=8, markeredgecolor='black')
            elif pattern.type.lower() == 'valley' and pattern.highlight_index in pattern.source_series.index:
                highlight_pos = index_to_position[pattern.highlight_index]
                plt_ax.plot(highlight_pos, pattern.source_series.loc[pattern.highlight_index],
                            'v', color=color, markersize=8, markeredgecolor='black')

        # Set x-ticks to show original index values
        if sorted_indices:
            PatternInterface.handle_sorted_indices(plt_ax, sorted_indices)

        # Set labels and title
        plt_ax.set_xlabel(patterns[0].index_name if patterns else 'Index')
        plt_ax.set_ylabel(patterns[0].value_name if patterns else 'Value')
        plt_ax.set_title(
            f"Multiple {patterns[0].type if patterns else 'Unimodality'} Patterns" if title is None else title)

        # Add legend
        plt_ax.legend()

        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # Ensure bottom margin for x-labels
        plt_ax.figure.subplots_adjust(bottom=0.15)

    def __init__(self, source_series: pd.Series, type: Literal['Peak', 'Valley'], highlight_index, value_name: str=None):
        """
        Initialize the UnimodalityPattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param type: The type of the pattern. Either 'Peak' or 'Valley' is expected.
        :param highlight_index: The index of the peak or valley.
        :param value_name: The name of the value to display.
        """
        self.source_series = source_series
        self.type = type
        self.highlight_index = highlight_index
        self.index_name = source_series.index.name if source_series.index.name else 'Index'
        self.value_name = value_name if value_name else 'Value'
        self.hash = None

    def visualize(self, plt_ax, title: str = None) -> None:
        """
        Visualize the unimodality pattern.
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name], title=None)
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
        return  (self.type == other.type and
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



class TrendPattern(PatternInterface):

    __name__ = "Trend pattern"

    @staticmethod
    def visualize_many(plt_ax, patterns: List['TrendPattern'], labels: List[str], title: str = None,
                       show_data: bool = True, alpha_data: float = 0.6) -> None:
        """
        Visualize multiple trend patterns on a single plot.

        :param plt_ax: Matplotlib axes to plot on
        :param patterns: List of TrendPattern objects
        :param labels: List of labels for each pattern
        :param title: Optional custom title for the plot
        :param show_data: Whether to show the raw data points (can be set to False if too cluttered)
        :param alpha_data: Opacity of the raw data (lower value reduces visual clutter)
        """
        # Define a color cycle for lines
        colors = plt.cm.tab10.colors

        # Define line styles for additional differentiation.
        # Taken from the matplotlib docs.
        line_styles = [
            ('loosely dotted', (0, (1, 10))),
            ('dotted', (0, (1, 5))),
            ('densely dotted', (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('loosely dashed', (0, (5, 10))),
            ('dashed', (0, (5, 5))),
            ('densely dashed', (0, (5, 1))),
            ('loosely dashdotted', (0, (3, 10, 1, 10))),
            ('dashdotted', (0, (3, 5, 1, 5))),
            ('densely dashdotted', (0, (3, 1, 1, 1))),
            ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

        # Prepare patterns with consistent numeric positions
        index_to_position, sorted_indices = PatternInterface.prepare_patterns_for_visualization(patterns)

        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)][1]

            # Map series to numeric positions for plotting
            x_positions = [index_to_position[idx] for idx in pattern.source_series.index]
            values = pattern.source_series.values

            # Plot the raw data with reduced opacity if requested
            if show_data:
                plt_ax.plot(x_positions, values, color=color, alpha=alpha_data, linewidth=1)

            # Plot the trend line using numeric positions
            trend_label = f"{label}"
            x_range = np.arange(len(sorted_indices))
            plt_ax.plot(x_range, pattern.slope * x_range + pattern.intercept,
                        linestyle=line_style, color=color, linewidth=2, label=trend_label)

        # Set x-ticks to show original index values
        if sorted_indices:
            PatternInterface.handle_sorted_indices(plt_ax, sorted_indices)

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

        default_title = f"Multiple Trend Patterns"
        plt_ax.set_title(title if title is not None else default_title)

        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # Add legend
        plt_ax.legend()

        # Ensure bottom margin for x-labels
        plt_ax.figure.subplots_adjust(bottom=0.15)

    def __init__(self, source_series: pd.Series, type: Literal['Increasing', 'Decreasing'],
                 slope: float, intercept: float = 0, value_name: str = None):
        """
        Initialize the Trend pattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param type: The type of the pattern.
        :param slope: The slope of the trend.
        """
        self.source_series = source_series
        self.type = type
        self.slope = slope
        self.intercept = intercept
        self.value_name = value_name if value_name else 'Value'
        self.hash = None

    def visualize(self, plt_ax, title: str = None) -> None:
        """
        Visualize the trend pattern.
        :param plt_ax:
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name], title=None)
        if title is not None:
            plt_ax.set_title(title)
        else:
            plt_ax.set_title(f"{self.type} trend in {self.value_name} with slope {self.slope:.2f} and intercept {self.intercept:.2f}")

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


class OutlierPattern(PatternInterface):

    __name__ = "Outlier pattern"

    @staticmethod
    def visualize_many(plt_ax, patterns: List['OutlierPattern'], labels: List[str], title: str = None,
                       show_regular: bool = True, alpha_regular: float = 0.5, alpha_outliers: float = 0.9) -> None:
        """
        Visualize multiple outlier patterns on a single plot.
        """
        colors = plt.cm.tab10.colors
        regular_marker = 'o'
        outlier_marker = 'X'

        # Prepare patterns with consistent numeric positions
        index_to_position, sorted_indices = PatternInterface.prepare_patterns_for_visualization(patterns)

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
            PatternInterface.handle_sorted_indices(plt_ax, sorted_indices)

        # Setup the rest of the plot
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], marker=outlier_marker, color='black',
                               markerfacecolor='black', markersize=10, linestyle='')]
        custom_labels = ['Outliers (marked with X)']

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

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
        self.source_series = source_series
        self.outlier_indexes = outlier_indexes
        self.outlier_values = outlier_values
        self.value_name = value_name if value_name else 'Value'
        self.hash = None

    def visualize(self, plt_ax, title: str = None) -> None:
        """
        Visualize the outlier pattern.
        :param plt_ax:
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name], title=None)
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


class CyclePattern(PatternInterface):

    __name__ = "Cycle pattern"

    @staticmethod
    def visualize_many(plt_ax, patterns: List['CyclePattern'], labels: List[str], title: str = None,
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
        self.source_series = source_series
        # Cycles is a dataframe with the columns: t_start, t_end, t_minimum, doc, duration
        self.cycles = cycles
        self.hash = None
        self._cycle_tuples = frozenset((row['t_start'], row['t_end']) for _, row in cycles.iterrows())
        self.value_name = value_name if value_name else 'Value'

    def visualize(self, plt_ax, title: str = None):
        """
        Visualize the cycle pattern.
        :param plt_ax:
        :return:
        """
        self.visualize_many(plt_ax, [self], [self.value_name], title=None, alpha_cycles=0.5, line_alpha=0.8)
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