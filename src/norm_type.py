
from numpy import *


class NormType:
    """
    Class with normalization types
    """

    @staticmethod
    def min_max_norm(data: ndarray, i: int, col_to_norm: int):
        """
        Useful when working with image compressing.
        Highly sensitive for data outliers.(High variance not accurate).
        This method can minimize the effect of outliers, because we end up with smaller variance.
        :param data: Data ndarray to normalize
        :param i: item index
        :param col_to_norm: column index
        :return: None
        """
        data[i, col_to_norm] = (data[i, col_to_norm] - min(data[:, col_to_norm])) / (max(data[:, col_to_norm]) - min(data[:, col_to_norm]))

    @staticmethod
    def stand_norm(data: ndarray, i: int, col_to_norm: int):
        """
        Method to standarize the data wit std = 1 and mean = 0.
        Use it when your data has a gaussian distribution.
        Can be useful for normalizing data for clustering.
        Does not provide gaussian dist for not gaussian input.
        Commonly used for all data analise problems.
        Without it, in gradient descent algorithms some weights can update faster than others.
        :return: None
        """
        data[i, col_to_norm] = (data[i, col_to_norm] - mean(data[:, col_to_norm])) / std(data[:, col_to_norm])

    @staticmethod
    def data_stand_norm(data: ndarray):
        """
        :param data:
        :return:
        """
        # for each column in data
        m, n = shape(data)
        for i in range(n):
            data[:, n] = NormType.row_stand_norm(data[:, n])


    @staticmethod
    def row_stand_norm(data: ndarray):
        """
        Normalization technique for one row data set.
        :return: normalized data list
        """
        return (data - mean(data)) / std(data)

    @staticmethod
    def unit_norm(data: ndarray, i: int, col_to_norm: int):
        data[i, col_to_norm] = (data[i, col_to_norm]) / max(data[:, col_to_norm])
