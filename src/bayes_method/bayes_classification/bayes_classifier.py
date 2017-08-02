
from src.math_oper import MathOper
from numpy import *
import operator
from src.data_manager import DataManager

class BayesClassifier:

    def _create_classes_skeleton(self, task_classes: list, categories: list, categorical_mask: list) -> dict:
        """
        :param task_classes: list with classes to predict
        :param categories: data items categories
        :param categorical_mask: list with items categories as bools
        :return: dict with initialized classes
        """
        classes = {}
        for task_class in task_classes:
            classes[task_class] = {}
            for category in categories:
                cat_index = categories.index(category)
                if categorical_mask[cat_index] == True:
                    classes[task_class][category] = {}
                else:
                    classes[task_class][category] = []
        return classes

    def _categorize_data(self, classes: dict, inputs: ndarray, target: ndarray, categories: list, categorical_mask: list) -> dict:
        """
        :param classes:
        :param inputs:
        :param target:
        :param categories:
        :param categorical_mask:
        :return: dict with categorized data items attached with classes
        """

        # first init classes with contionus data
        # we do not have to go through all data
        # we can order the data
        # so order the data

        task_classes = [0, 1]
        ordered_data = DataManager.order_data(inputs, target, task_classes)

        # categorize continous data
        for task_class in task_classes:
            for category in categories:
                cat_index = categories.index(category)
                # contious data
                if categorical_mask[cat_index] == False:
                   # compute data feature probs
                   feature_prob = BayesClassifier.compute_feature_prop(ordered_data[task_class][:, cat_index],8)
                   classes[task_class][category] = feature_prob

        # categorize labeled data
        for index, item in enumerate(inputs):
            # each item is a row
            for category in categories:
                # for categorical item
                cat_index = categories.index(category)
                if categorical_mask[cat_index] == True:
                    if item[cat_index] in classes[target[index]][categories[cat_index]]:
                        classes[target[index]][categories[cat_index]][item[cat_index]] += 1
                    else:
                        classes[target[index]][categories[cat_index]][item[cat_index]] = 0
        return classes

    def init_classes(self, inputs: ndarray, target: ndarray, categories: list, categorical_mask: list, task_classes: list) -> dict:
        """
        :param inputs: ndarray with data inputs
        :param target: ndarray with data targets
        :param categories: list with all inputs categories
        :param categorical_mask: list with categories masks
        :param task_classes: Problem available classes
        :return: dict with categorized data and adjusted probability.
        """
        classes = self._create_classes_skeleton(task_classes, categories, categorical_mask)
        classes = self._categorize_data(classes, inputs, target, categories, categorical_mask)

        # compute data probability
        for class_item in classes:
            for index, category in enumerate(classes[class_item]):
                if categorical_mask[index] == True:
                    class_cate_vals = classes[class_item][category]
                    cate_points, cate_props = MathOper.get_data_prop(class_cate_vals, 100)
                    classes[class_item][category] = dict(zip(cate_props, cate_points))
                else:
                    class_cate_sum_val = 0
                    for cate_value in classes[class_item][category]:
                        class_cate_sum_val += classes[class_item][category][cate_value]
                    for cate_item in classes[class_item][category]:
                        cate_item_value = classes[class_item][category][cate_item]
                        classes[class_item][category][cate_item] = cate_item_value / class_cate_sum_val
        return classes


    def prepare_test_items(self, items: ndarray, categories: list) -> list:
        """
        Transform the data in ndarray to dict format.
        :param items:
        :param categories: list with items categories
        :return: categorized item dict.
        """
        items_dicts = []
        for item in items:
            item_dict = dict(zip(categories, item))
            items_dicts.append(item_dict)
        return items_dicts


    def test_classify(self, classes: dict, test_inputs: ndarray, test_targets: ndarray, class_props: dict):
        # for each test item
        right_answeres = 0
        est_classes = []
        for index, test_input in enumerate(test_inputs):
            # classify item
            est_class = self.classify_item(test_input, classes, class_props)
            est_classes.append(est_class)
            if est_class == test_targets[index]:
                right_answeres += 1
        correctness = right_answeres / len(test_inputs)
        return correctness, est_classes

    # get probability of the closest item in the list.
    def get_item_prop(self, item_value: float, item_values: dict):
        """

        :param item_value: float value
        :param item_values:
        :return:
        """
        if len(item_values) == 0:
            raise ValueError("Empty dict error")
        # implement quick search
        keys = list(item_values.keys())
        list_to_search = copy(keys)
        while True:
            N = len(list_to_search)
            if N == 1:
                return item_values[list_to_search[0]]
            d = N / 2
            d = int(d)
            one = list_to_search[0:d]
            two = list_to_search[d:N]
            if item_value >= two[0]:
                list_to_search = two
            else:
                list_to_search = one


    def classify_item(self, item: dict, classes: dict, class_props: dict) -> float:
        """
        :param item: Item to classify
        :param classes:
        :return: estimated class
        """
        est_props = {}
        for class_item in classes:
            prop_sum = 0
            for category in classes[class_item]:
                item_cat_value = item[category]
                if type(item_cat_value) == float:
                        category_prop = classes[class_item][category]
                        one = self.get_item_prop(item_cat_value, category_prop)
                        two = log(class_props[class_item])
                        prop_sum += one + two
                else:
                    if item[category] not in classes[class_item][category]:
                        print("category: {} not in class: {}".format(category, class_item))
                        continue
                    prop_sum += classes[class_item][category][item[category]]
            est_props[class_item] = prop_sum # + log(class_props[class_item])
        result = max(est_props.items(), key=operator.itemgetter(1))[0]
        return result

    """
    This is our new Approach
    """
    @staticmethod
    def compute_feature_prop(data: ndarray, sections: int):
        if data.ndim != 1:
            raise ValueError("Data is not a vector.")
        feature_prop = MathOper.get_prop_data(data, sections)
        return feature_prop

    @staticmethod
    def class_feature_diff(feature_class_1: ndarray, feature_class_2: ndarray, sections: int):
        """
        Computes props diff between features
        :return:
        """
        if feature_class_1.ndim != 1 or feature_class_2.ndim != 1:
            raise ValueError("Feature data ndarrays are not vectors.")
        feature_props_1 = BayesClassifier.compute_feature_prop(feature_class_1, sections)
        feature_props_2 = BayesClassifier.compute_feature_prop(feature_class_2, sections)
        prop_diff = 0
        for i in range(len(feature_props_1)):
            margin = abs(feature_props_1[i][1] - feature_props_2[i][1])
            prop_diff += margin
        return prop_diff

    @staticmethod
    def compute_item_feature_fit(item_value, feature_props: ndarray):
        """
        Gets probability of item_value fit to this feature.
        If item_value is not in range of feature_props then returns 0.
        :param item_value:
        :param feature_props: list with tuples.
        :return:
        """
        for feature_prop in feature_props:
            down = feature_prop[0][0]
            up = feature_prop[0][1]
            if item_value >= down and item_value <= up:
                return feature_prop[1]
        return 0


