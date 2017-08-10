
from numpy import *


class Normalizer:
    """
    Generic data normalizer
    """

    @staticmethod
    def normalize(data: ndarray, norm_type, cols_to_norm: list = None):
        """
        :param data: Data ndarray to normalize
        :param norm_type: Normalization type
        :param cols_to_norm: List with column indexes to normalize. If empty then normalize all data ndarray.
        :return: Normalized data ndarray
        """
        data = array(data)
        m, n = shape(data)
        if cols_to_norm == None:
            cols_to_norm = range(0, n)
        for col_to_norm in cols_to_norm:
            if col_to_norm < min(cols_to_norm) or col_to_norm > max(cols_to_norm):
                print("Column index not in range: {}".format(col_to_norm))
                continue
            norm_type(data, col_to_norm)
        return data
