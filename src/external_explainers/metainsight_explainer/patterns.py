from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

class PatternInterface(ABC):
    """
    Abstract base class for defining patterns.
    """

    @abstractmethod
    def visualize(self, plt_ax) -> None:
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


class UnimodalityPattern(PatternInterface):

    def __init__(self, source_series: pd.Series, type: Literal['Peak', 'Valley'], highlight_index):
        """
        Initialize the UnimodalityPattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param type: The type of the pattern. Either 'Peak' or 'Valley' is expected.
        :param highlight_index: The index of the peak or valley.
        :param index_name: The name of the index.
        """
        self.source_series = source_series
        self.type = type
        self.highlight_index = highlight_index
        self.index_name = source_series.index.name if source_series.index.name else 'Index'
        self.hash = None

    def visualize(self, plt_ax) -> None:
        """
        Visualize the unimodality pattern.
        :return:
        """
        plt_ax.plot(self.source_series)
        plt_ax.set_xlabel(self.index_name)
        plt_ax.set_ylabel('Value')
        # Emphasize the peak or valley
        if self.type.lower() == 'peak':
            plt_ax.plot(self.highlight_index, self.source_series[self.highlight_index], 'ro', label='Peak')
        elif self.type.lower() == 'valley':
            plt_ax.plot(self.highlight_index, self.source_series[self.highlight_index], 'bo', label='Valley')
        plt_ax.legend()


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

    def __init__(self, source_series: pd.Series, type: Literal['Increasing', 'Decreasing'],
                 slope: float, intercept: float = 0):
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
        self.hash = None

    def visualize(self, plt_ax) -> None:
        """
        Visualize the trend pattern.
        :param plt_ax:
        :return:
        """
        plt_ax.plot(self.source_series)
        plt_ax.set_xlabel(self.source_series.index.name if self.source_series.index.name else 'Index')
        plt_ax.set_ylabel('Value')
        x_numeric = np.arange(len(self.source_series))
        # Emphasize the trend
        plt_ax.plot(self.source_series.index, self.slope * x_numeric + self.intercept, 'g--',
                    linewidth=2,
                    label='Increasing Trend' if self.type.lower() == 'Increasing' else 'Decreasing Trend')
        plt_ax.legend()

    def __eq__(self, other) -> bool:
        """
        Check if two TrendPattern objects are equal.
        :param other: Another TrendPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if they have the same type
        (increasing / decreasing) (we trust that comparisons will be done on the same series).
        """
        if not isinstance(other, TrendPattern):
            return False
        # We do not compare the slope and intercept - we only ca
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

    def __init__(self, source_series: pd.Series, outlier_indexes: pd.Index, outlier_values: pd.Series):
        """
        Initialize the Outlier pattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param outlier_indexes: The indexes of the outliers.
        :param outlier_values: The values of the outliers.
        """
        self.source_series = source_series
        self.outlier_indexes = outlier_indexes
        self.outlier_values = outlier_values
        self.hash = None

    def visualize(self, plt_ax) -> None:
        """
        Visualize the outlier pattern.
        :param plt_ax:
        :return:
        """
        plt_ax.scatter(self.source_series.index, self.source_series, label='Regular Data Point')
        plt_ax.set_xlabel(self.source_series.index.name if self.source_series.index.name else 'Index')
        plt_ax.set_ylabel('Value')
        # Emphasize the outliers
        plt_ax.scatter(self.outlier_indexes, self.outlier_values, color='red', label='Outliers')
        plt_ax.legend()


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

    def __init__(self, source_series: pd.Series, cycles: pd.DataFrame):
        """
        Initialize the Cycle pattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param cycles: The cycles detected in the series.
        """
        self.source_series = source_series
        # Cycles is a dataframe with the columns: t_start, t_end, t_minimum, doc, duration
        self.cycles = cycles
        self.hash = None

    def visualize(self, plt_ax):
        """
        Visualize the cycle pattern.
        :param plt_ax:
        :return:
        """
        plt_ax.plot(self.source_series)
        plt_ax.set_xlabel(self.source_series.index.name if self.source_series.index.name else 'Index')
        plt_ax.set_ylabel('Value')
        i = 1
        # Emphasize the cycles, and alternate colors
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        color_index = 0
        for _, cycle in self.cycles.iterrows():
            plt_ax.axvspan(cycle['t_start'], cycle['t_end'], color=colors[color_index], alpha=0.5, label=f'Cycle {i}')
            i += 1
            color_index = (color_index + 1) % len(colors)
        plt_ax.legend()

    def __eq__(self, other):
        """
        Check if two CyclePattern objects are equal.
        :param other:
        :return: True if they are equal, False otherwise. They are considered equal if the cycles of one are a
        subset of the other or vice versa.
        """
        if not isinstance(other, CyclePattern):
            return False
        return self.cycles.isin(other.cycles).all().all() or \
                other.cycles.isin(self.cycles).all().all()

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