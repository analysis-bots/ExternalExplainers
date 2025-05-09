from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    def __init__(self, source_series: pd.Series, type: str, index, index_name: str = None):
        """
        Initialize the UnimodalityPattern with the provided parameters.

        :param source_series: The source series to evaluate.
        :param type: The type of the pattern.
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



if __name__ == '__main__':
    data = np.random.normal(0, 1, 1000)
    series = pd.Series(data)
    pattern = UnimodalityPattern(series, 'Peak', 500)
    fig, ax = plt.subplots()
    pattern.visualize(ax)
    plt.show()