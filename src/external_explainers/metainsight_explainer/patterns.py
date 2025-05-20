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


class UnimodalityPattern(PatternInterface):

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

        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]

            # Plot the series with a unique color
            plt_ax.plot(pattern.source_series, color=color, alpha=0.7, label=label)

            # Highlight the peak or valley with a marker
            if pattern.type.lower() == 'peak':
                plt_ax.plot(pattern.highlight_index, pattern.source_series[pattern.highlight_index],
                            'o', color=color, markersize=8, markeredgecolor='black')
            elif pattern.type.lower() == 'valley':
                plt_ax.plot(pattern.highlight_index, pattern.source_series[pattern.highlight_index],
                            'v', color=color, markersize=8, markeredgecolor='black')

        # Set labels and title
        plt_ax.set_xlabel(patterns[0].index_name if patterns else 'Index')
        plt_ax.set_ylabel(patterns[0].value_name if patterns else 'Value')
        plt_ax.set_title(f"Multiple {patterns[0].type if patterns else 'Unimodality'} Patterns" if title is None else title)

        # Add legend outside the plot
        plt_ax.legend()

        #plt_ax.figure.subplots_adjust(right=0.5)  # Reserve 50% of width for legend

        # Rotate x-axis tick labels if needed
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

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
        plt_ax.plot(self.source_series)
        plt_ax.set_xlabel(self.index_name)
        plt_ax.set_ylabel(self.value_name)
        # Emphasize the peak or valley
        if self.type.lower() == 'peak':
            plt_ax.plot(self.highlight_index, self.source_series[self.highlight_index], 'ro', label='Peak')
        elif self.type.lower() == 'valley':
            plt_ax.plot(self.highlight_index, self.source_series[self.highlight_index], 'bo', label='Valley')
        plt_ax.legend(loc="upper left")
        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        if title is not None:
            plt_ax.set_title(title)


    def __eq__(self, other) -> bool:
        """
        Check if two UnimodalityPattern objects are equal.
        :param other: Another UnimodalityPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if they have the same type,
        the same highlight index.
        """
        if not isinstance(other, UnimodalityPattern):
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

        # Define line styles for additional differentiation. This is taken from matplotlib's
        # docs: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
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

        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]
            line_style = line_styles[i  % len(line_styles)][1]

            # Plot the raw data with reduced opacity if requested
            if show_data:
                plt_ax.plot(pattern.source_series, color=color, alpha=alpha_data, linewidth=1)

            # Get x range for trend line
            x_numeric = np.arange(len(pattern.source_series))

            # Plot the trend line
            trend_label = f"{label}"
            plt_ax.plot(pattern.source_series.index, pattern.slope * x_numeric + pattern.intercept,
                        linestyle=line_style, color=color, linewidth=2, label=trend_label)

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

        default_title = f"Multiple Trend Patterns"
        plt_ax.set_title(title if title is not None else default_title)

        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # First, adjust the subplot parameters to make room for the legend
        #plt_ax.figure.subplots_adjust(right=0.5)  # Reserve 50% of width for legend

        # Place legend outside the plot
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
        plt_ax.plot(self.source_series)
        plt_ax.set_xlabel(self.source_series.index.name if self.source_series.index.name else 'Index')
        plt_ax.set_ylabel(self.value_name)
        x_numeric = np.arange(len(self.source_series))
        # Emphasize the trend
        label = f"y={self.slope:.2f}x + {self.intercept:.2f}"
        plt_ax.plot(self.source_series.index, self.slope * x_numeric + self.intercept, 'g--',
                    linewidth=2,
                    label=label)
        plt_ax.legend(loc="upper left")
        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        if title is not None:
            plt_ax.set_title(title)

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

    @staticmethod
    def visualize_many(plt_ax, patterns: List['OutlierPattern'], labels: List[str], title: str = None,
                       show_regular: bool = True, alpha_regular: float = 0.5, alpha_outliers: float = 0.9) -> None:
        """
        Visualize multiple outlier patterns on a single plot.

        :param plt_ax: Matplotlib axes to plot on
        :param patterns: List of OutlierPattern objects
        :param labels: List of labels for each pattern
        :param title: Optional custom title for the plot
        :param show_regular: Whether to show regular (non-outlier) data points
        :param alpha_regular: Opacity for regular data points
        :param alpha_outliers: Opacity for outlier points
        """
        # Define a color cycle for different datasets
        colors = plt.cm.tab10.colors

        # Define marker styles
        regular_marker = 'o'  # Circle for regular points
        outlier_marker = 'X'  # X mark for outliers

        # Create a legend handle for the outlier explanation
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], marker=outlier_marker, color='black',
                               markerfacecolor='black', markersize=10, linestyle='')]
        custom_labels = ['Outliers (marked with X)']

        # Plot each dataset
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]

            # Plot regular data points if requested
            if show_regular:
                plt_ax.scatter(
                    pattern.source_series.index,
                    pattern.source_series,
                    color=color,
                    alpha=alpha_regular,
                    marker=regular_marker,
                    s=30,  # Size
                    label=label
                )
            else:
                # Still add to legend even if not showing points
                plt_ax.scatter([], [], color=color, marker=regular_marker, s=30, label=label)

            # Plot outliers with the same color but a different marker
            if pattern.outlier_indexes is not None and len(pattern.outlier_indexes) > 0:
                plt_ax.scatter(
                    pattern.outlier_indexes,
                    pattern.outlier_values,
                    color=color,
                    alpha=alpha_outliers,
                    marker=outlier_marker,
                    s=100,  # Larger size for outliers
                    edgecolors='black',  # Black edge for visibility
                    linewidth=1.5
                )

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

        default_title = "Multiple Outlier Patterns"
        plt_ax.set_title(title if title is not None else default_title)

        # Rotate x-axis tick labels if needed
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # Get the current handles and labels
        handles, labels_current = plt_ax.get_legend_handles_labels()

        # Combine with custom outlier explanation
        all_handles = handles + custom_lines
        all_labels = labels_current + custom_labels

        # Adjust subplot parameters to make room for the legend
        #plt_ax.figure.subplots_adjust(right=0.5)  # Reserve 30% of width for legend

        # Place legend outside the plot with combined handles/labels
        plt_ax.legend(all_handles, all_labels)

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
        plt_ax.scatter(self.source_series.index, self.source_series, label='Regular Data Point')
        plt_ax.set_xlabel(self.source_series.index.name if self.source_series.index.name else 'Index')
        plt_ax.set_ylabel(self.value_name)
        # Emphasize the outliers
        plt_ax.scatter(self.outlier_indexes, self.outlier_values, color='red', label='Outliers')
        plt_ax.legend(loc="upper left")
        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        if title is not None:
            plt_ax.set_title(title)


    def __eq__(self, other):
        """
        Check if two OutlierPattern objects are equal.
        :param other: Another OutlierPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if the index set of one is a subset
        of the other or vice versa.
        """
        if not isinstance(other, OutlierPattern):
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
        import pandas as pd

        # Define a color cycle for lines
        colors = plt.cm.tab10.colors

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
                    all_cycle_data.append((cycle['t_start'], cycle['t_end']))

        # Find common cycle periods
        common_periods = []
        if len(patterns) > 1 and all_cycle_data:
            # Handle datetime objects by creating a time_points array differently
            # Get all unique timestamps from starts and ends
            all_timestamps = sorted(list(set([t for start, end in all_cycle_data for t in [start, end]])))

            # Create additional points between timestamps if needed
            if len(all_timestamps) > 1:
                time_points = []
                for i in range(len(all_timestamps) - 1):
                    # Add the current timestamp
                    time_points.append(all_timestamps[i])

                    # Add intermediate points if the gap is large enough
                    curr = pd.Timestamp(all_timestamps[i])
                    next_ts = pd.Timestamp(all_timestamps[i + 1])
                    if (next_ts - curr).total_seconds() > 60:  # If gap is more than a minute
                        # Add 10 intermediate points
                        delta = (next_ts - curr) / 11
                        for j in range(1, 11):
                            time_points.append(curr + delta * j)

                # Add the last timestamp
                time_points.append(all_timestamps[-1])
            else:
                time_points = all_timestamps

            # For each time point, check if it falls within a cycle for each pattern
            overlap_counts = np.zeros(len(time_points))

            for pattern in patterns:
                if hasattr(pattern, 'cycles') and not pattern.cycles.empty:
                    pattern_mask = np.zeros(len(time_points), dtype=bool)
                    for _, cycle in pattern.cycles.iterrows():
                        start, end = cycle['t_start'], cycle['t_end']
                        pattern_mask = pattern_mask | (
                                    (np.array(time_points) >= start) & (np.array(time_points) <= end))
                    overlap_counts += pattern_mask

            # Find regions where all patterns have a cycle
            common_mask = overlap_counts == len(patterns)

            # Find contiguous regions of common cycles
            if np.any(common_mask):
                changes = np.diff(np.concatenate(([0], common_mask.astype(int), [0])))
                start_indices = np.where(changes == 1)[0]
                end_indices = np.where(changes == -1)[0] - 1

                for start_idx, end_idx in zip(start_indices, end_indices):
                    common_periods.append((time_points[start_idx], time_points[end_idx]))

        # Plot each pattern
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            color = colors[i % len(colors)]

            # Plot the time series
            line, = plt_ax.plot(pattern.source_series, color=color, alpha=line_alpha, linewidth=2)
            legend_handles.append(line)
            legend_labels.append(label)

            # Highlight each cycle with a semi-transparent fill
            if hasattr(pattern, 'cycles') and not pattern.cycles.empty:
                # Add individual cycle legend element
                cycle_patch = plt.Rectangle((0, 0), 1, 1, color=color, alpha=alpha_cycles)

                for _, cycle in pattern.cycles.iterrows():
                    # Highlight the cycle only if it is not in the common cycles - we highlight those later.
                    if not any(
                            start <= cycle['t_start'] <= end and start <= cycle['t_end'] <= end
                            for start, end in common_periods
                    ):
                        t_start = cycle['t_start']
                        t_end = cycle['t_end']

                        # Highlight the cycle region
                        plt_ax.axvspan(t_start, t_end, color=color, alpha=alpha_cycles)

        # Highlight common cycles
        if common_periods:
            for start, end in common_periods:
                plt_ax.axvspan(start, end, color=common_cycle_color, alpha=alpha_cycles * 1.5, zorder=-1)

            # Add legend item for common cycles
            common_patch = plt.Rectangle((0, 0), 1, 1, color=common_cycle_color, alpha=alpha_cycles * 1.5)
            legend_handles.append(common_patch)
            legend_labels.append('Common cycles (all patterns)')

        # Set labels and title
        if patterns:
            plt_ax.set_xlabel(patterns[0].source_series.index.name if patterns[0].source_series.index.name else 'Index')
            plt_ax.set_ylabel(patterns[0].value_name if patterns[0].value_name else 'Value')

        default_title = "Multiple Cycle Patterns"
        plt_ax.set_title(title if title is not None else default_title)

        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)

        # Adjust subplot parameters to make room for the legend
        #plt_ax.figure.subplots_adjust(right=0.5)  # Reserve 50% of width for legend

        # Place legend outside the plot
        plt_ax.legend(legend_handles, legend_labels)

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
        plt_ax.plot(self.source_series)
        plt_ax.set_xlabel(self.source_series.index.name if self.source_series.index.name else 'Index')
        plt_ax.set_ylabel(self.value_name)
        i = 1
        # Emphasize the cycles, and alternate colors
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        color_index = 0
        for _, cycle in self.cycles.iterrows():
            plt_ax.axvspan(cycle['t_start'], cycle['t_end'], color=colors[color_index], alpha=0.5, label=f'Cycle {i}')
            i += 1
            color_index = (color_index + 1) % len(colors)
        plt_ax.legend(loc="upper left")
        # Rotate x-axis tick labels
        plt.setp(plt_ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        if title is not None:
            plt_ax.set_title(title)

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