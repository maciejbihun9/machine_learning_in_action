
from numpy import *

class ClusterClassifier:

    @staticmethod
    def get_array_mean():

    @staticmethod
    def get_class_cluster(class_item: ndarray):
        """
        Computes cluster params for given class.
        :param class_item: Contains class items
        :return: ndarray with cluster params.
        """
        m,n = shape(class_item)
        cluster_params = array([0.0] * n)
        for category in range(n):
            # get category mean value
            cate_mean = mean(class_item[category])



    @staticmethod
    def get_item_probs(classes: ndarray, item: ndarray):
        """
        :param classes:
        :param item:
        :return:
        """
        for class_item in classes:
            # for each row in class
            for row in class_item:
                dist = ClusterClassifier.get_item_dist(row, item)

    @staticmethod
    def get_item_dist(cluster_items: ndarray, item: ndarray):
        """
        Computes the distance between item and cluster.
        :param cluster_items: ndarray with cluster properties
        :param item: ndarray with item properties
        :return: float distance between item and cluster
        """
        sum = 0
        for item_index, cluster_item in enumerate(cluster_items):
            sum += pow(item[item_index] - cluster_items[item_index], 2)
        return sqrt(sum)
