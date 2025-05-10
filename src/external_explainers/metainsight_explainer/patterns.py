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
    def visualize(self, ax):
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

    def __init__(self, source_series: pd.Series, type: Literal['Peak', 'Valley'], index, index_name: str = None):
        """
        Initialize the UnimodalityPattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param type: The type of the pattern. Either 'Peak' or 'Valley' is expected.
        :param location: The location of the pattern.
        """
        self.source_series = source_series
        self.type = type
        self.index = index
        self.index_name = index_name

    def visualize(self, ax):
        """
        Visualize the unimodality pattern.
        :return:
        """
        ax.plot(self.source_series, label='Unimodality Pattern')
        ax.axvline(x=self.index, color='r', linestyle='--', label='Location')
        ax.legend()
        ax.set_title(f'Unimodality Pattern: {self.type}')


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
        :return: True if they are equal, False otherwise.
        """
        if not isinstance(other, UnimodalityPattern):
            return False
        return (self.source_series.equals(other.source_series) and
                self.type == other.type and
                self.index == other.index and
                self.index_name == other.index_name)



class TrendPattern(PatternInterface):

    def __init__(self, source_series: pd.Series, type: Literal['Increasing', 'Decreasing'], slope: float, intercept: float = 0):
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

    def visualize(self, ax):
        pass

    def create_explanation_string(self, commonness_set, exceptions):
        pass

    def __eq__(self, other):
        pass


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

    def visualize(self, ax):
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

    def visualize(self, ax):
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