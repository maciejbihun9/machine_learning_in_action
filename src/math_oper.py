
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
    def get_prop_data(data: ndarray, parts: int, min_val: float, max_val: float) -> ndarray:
        """
        Parts is computed using range value.
        :param data:
        :param parts:
        :return:
        """
        data = sorted(data)

        length = len(data)

        margin = max_val - min_val

        # establish on how many sections you want to split your data using number of items.
        interval = margin / parts

        sections = []
        cur_min = min_val
        # create sections
        for i in range(parts):
            sections.append((cur_min, cur_min + interval))
            cur_min = cur_min + interval

        for section in sections:
            counter = 0
            for item in data:
                if item >= section[0] and item < section[1]:
                    counter += 1
                elif section == sections[-1] and item == max_val:
                    counter += 1
            sections[sections.index(section)] = (section, round(counter / length, 4))
        return sections


    @staticmethod
    def get_classes_prop(target: ndarray, class_categories: list) -> dict:
        """
        :param target: ndarray with all target classes
        :param class_categories: class categories
        :return: dict with class category selection probability
        """
        length = len(target)
        props = {}
        target = target.tolist()
        for class_item in class_categories:
            counter = target.count(class_item)
            props[class_item] = counter / length
        return props

