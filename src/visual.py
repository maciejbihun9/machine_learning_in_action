
import matplotlib.pyplot as plt
from numpy import *

class Visual:

    @staticmethod
    def plot_hist(data: ndarray):
        mean_val = mean(data)
        std_val = std(data)
        bludgeon = int(len(data) / 100)
        min = mean_val - 5 * std_val
        max = mean_val + 5 * std_val
        points, props, patches = plt.hist(data, bludgeon , normed=1, facecolor='g', alpha=0.75)

        plt.axis([min, max, 0, 0.03])
        plt.grid(True)
        plt.show()

