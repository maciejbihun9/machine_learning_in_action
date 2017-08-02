
from numpy import *
import matplotlib.pyplot as plt

class MathOper:

    @staticmethod
    def values_between(data_array: ndarray, min: float, max: float) -> ndarray:
        """
        Get array with values that indicates weather are between min and max
        :param data_array: data array to check(Size does not metter)
        :param min: Minimal value
        :param max: Maximal value
        :return: True/False ndarray
        """
        return logical_and(data_array >= min, data_array <= max)



    @staticmethod
    def get_data_prop(data: ndarray, parts: int):
        """
        :param data: ndarray of data
        :param parts: divide data on parts
        :return: ndarray with probabilities of selecting an items
        """
        points, props, patches = plt.hist(data, parts, normed=1, facecolor='g', alpha=0.75)
        # get first item index which variable has better value than item
        return points, props

    @staticmethod
    def get_prop_data(data: ndarray, parts: int) -> ndarray:
        """
        Parts is computed using range value.
        :param data:
        :param parts:
        :return:
        """
        data = sorted(data)

        length = len(data)
        props = []

        min_val = min(data)
        max_val = max(data)
        margin = max_val - min_val

        # establish on how many sections you want to split your data using number of items.
        interval = margin / parts

        # starting values
        start = min_val
        end = min_val + interval

        counter = 0
        for item in data:
            if item >= start and item <= end:
                counter += 1
            if item >= end:
                value = round(counter / length, 4)
                props.append(((start, end), value))
                counter = 0
                start = end
                end = end + interval
                continue
        return props




    @staticmethod
    def get_classes_prop(classes: ndarray, class_categories: list) -> dict:
        """
        :param classes: ndarray with all target classes
        :param class_categories: class categories
        :return: dict with class category selection probability
        """
        length = len(classes)
        props = {}
        classes = classes.tolist()
        for class_item in class_categories:
            counter = classes.count(class_item)
            props[class_item] = counter / length
        return props

