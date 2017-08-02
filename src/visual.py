
import matplotlib.pyplot as plt
from numpy import *
from scipy.interpolate import UnivariateSpline


class Visual:

    @staticmethod
    def plot_prop_dist(data: ndarray, size: int):
        """
        Plot data probability distribution
        :param data:
        :return: None
        """
        data_dim = data.ndim
        if data_dim != 1:
            raise ValueError("Data array is not a vector")
        n = len(data)
        p, x = histogram(data, bins=size)  # bin it into n = N/10 bins
        x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
        f = UnivariateSpline(x, p, s=size)
        plt.plot(x, f(x))
        plt.show()

