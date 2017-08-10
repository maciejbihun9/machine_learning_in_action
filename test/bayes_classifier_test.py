import unittest

from numpy import *

from src.bayes.bayes_classifier import BayesClassifier
from src.data_manager import DataManager
from src.math_oper import MathOper


class BayesClassifierTest(unittest.TestCase):

    def setUp(self):
        url = '../resources/50k.txt'
        data = DataManager.load_data(url, False, True, ', ')
        data = array(data, dtype='object')

        # filter
        no_item_sign = '?'
        data = DataManager.data_filter(data, no_item_sign)

        N = 1000
        test_N = 200

        self.categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]

        self.inputs = data[0:N, 0:14]

        self.task_classes = [0, 1]

        target = data[0:N, 14]
        self.target = array([0 if '<=50' in y else 1 for y in target])

        self.test_target = data[0:test_N]
        self.test_inputs = data[0:test_N]

        self.class_props = MathOper.get_classes_prop(target, self.task_classes)

        self.bayesClassifier = BayesClassifier()
        self.data = self.bayesClassifier.init_classes(self.inputs, self.target, self.categories, categorical_mask, self.task_classes)

        ordered_data = DataManager.order_data(self.inputs, self.target, [0, 1])
        self.class_0 = ordered_data[0]
        self.class_1 = ordered_data[1]
    """
    def test_prepare_test_items(self):
        test_items = self.bayesClassifier.prepare_test_items(self.inputs, self.categories)

        # check if all items are dicts
        self.assertTrue(type(test_items) == list)

        print(test_items)
    """
    """
    # TESTED
    def test_class_feature_diff(self):

        feat_diff = self.bayesClassifier.class_feature_diff(self.class_0[:, 12], self.class_1[:, 12], 8)
        print("Feature diff: {}".format(feat_diff))
    """


    def test_compute_item_feature_fit(self):
        item_value = 45
        feat_prop = self.bayesClassifier.compute_feature_prop(self.class_0[:, 0], 8)
        item_fit = self.bayesClassifier.compute_item_feature_fit(item_value, feat_prop)

        print(item_fit)

if __name__ == '__main__':
    unittest.main()
