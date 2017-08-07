
import unittest
from src.math_oper import MathOper
from numpy import *
from src.data_manager import DataManager
from scipy.interpolate import UnivariateSpline
from src.normalizer import Normalizer
from src.norm_type import NormType
import matplotlib.pyplot as plt

class MathOperTest(unittest.TestCase):

    def setUp(self):
        url = '../resources/50k.txt'
        data = DataManager.load_data(url, False, True, ', ')
        data = array(data, dtype='object')

        # filter
        no_item_sign = '?'
        data = DataManager.data_filter(data, no_item_sign)
        N = 1000
        inputs = data[0:N, 0:14]
        self.test_data = inputs

    """
    def test_get_prop_data(self):
        test_data = array(self.test_data)
        test = MathOper.get_prop_data(test_data)
        print(test)
    """

    def test_get_norm_prob(self):
        # norm_test_data = Normalizer.normalize(self.test_data, NormType.min_max_norm, [2])
        norm_test_data = self.test_data[:, 2]
        # data_probs = MathOper.get_norm_prob(norm_test_data)
        test_data_mean = mean(norm_test_data)
        test_data_std = std(norm_test_data)
        test_data_prob_dist = random.normal(test_data_mean, test_data_std, 1000)
        plt.hist(test_data_prob_dist, 50)
        plt.show()


    """
    def test_prop_hist(self):
        n = len(self.test_data)
        p, x = histogram(self.test_data, bins=5)  # bin it into n = N/10 bins
        x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
        f = UnivariateSpline(x, p, s=5)
        plt.plot(x, f(x))
        plt.show()
    """

if __name__ == '__main__':
    unittest.main()
