"""
A simple util script for downloading the datasets used in the tests.
"""

from tests.utils import get_dataset

if __name__ == '__main__':
    get_dataset('adult')
    get_dataset('spotify')
    get_dataset('bank_churners')
    get_dataset('houses')