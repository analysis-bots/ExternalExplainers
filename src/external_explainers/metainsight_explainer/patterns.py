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
    def visualize(self, plt_ax):
        """
        Visualize the pattern.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def create_explanation_string(self, commonness_set, exceptions):
        """
        Create an explanation string for the pattern.
        :param commonness_set: The commonness set, where the pattern is common.
        :param exceptions: The exceptions dict, where the pattern is different from the commonness set or did not occur.
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def __eq__(self, other):
        """
        Check if two patterns are equal
        :param other: Another pattern of the same type
        :return:
        """


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

    def visualize(self, plt_ax):
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
        plt_ax.set_title(f'Unimodality Pattern: {self.type}')


    def create_explanation_string(self, commonness_set, exceptions):
        """
        Create an explanation string for the unimodality pattern.
        :param commonness_set: The commonness set, where the pattern is common.
        :param exceptions: The exceptions dict, where the pattern is different from the commonness set or did not occur.
        :return:
        """
        return f"Unimodality Pattern: {commonness_set}, Exceptions: {exceptions}"

    def __eq__(self, other):
        """
        Check if two UnimodalityPattern objects are equal.
        :param other: Another UnimodalityPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if they have the same type,
        the same highlight index, and are on the same index.
        """
        if not isinstance(other, UnimodalityPattern):
            return False
        return  (self.type == other.type and
                self.highlight_index == other.highlight_index and
                self.source_series.index == other.source_series.index)



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

    def visualize(self, plt_ax):
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
        plt_ax.set_title(f'Trend Pattern: {self.type}')

    def create_explanation_string(self, commonness_set, exceptions):
        pass

    def __eq__(self, other):
        """
        Check if two TrendPattern objects are equal.
        :param other: Another TrendPattern object.
        :return: True if they are equal, False otherwise. They are considered equal if they have the same type
        (increasing / decreasing) and are on the same index.
        """
        if not isinstance(other, TrendPattern):
            return False
        # We do not compare the slope and intercept - we only ca
        return self.source_series.index == other.source_series.index and \
                self.type == other.type


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

    def visualize(self, plt_ax):
        pass

    def create_explanation_string(self, commonness_set, exceptions):
        pass

    def __eq__(self, other):
        pass


class CyclePattern(PatternInterface):

    def __init__(self, source_series: pd.Series, cycles: pd.DataFrame):
        """
        Initialize the Cycle pattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param cycle_length: The length of the cycle.
        """
        self.source_series = source_series
        self.cycles = cycles

    def visualize(self, plt_ax):
        pass

    def create_explanation_string(self, commonness_set, exceptions):
        pass

    def __eq__(self, other):
        pass


if __name__ == '__main__':
    data = np.random.normal(0, 1, 1000)
    series = pd.Series(data)
    pattern = UnimodalityPattern(series, 'Peak', 500)
    fig, ax = plt.subplots()
    pattern.visualize(ax)
    plt.show()