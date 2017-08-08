
from src.math_oper import MathOper
from numpy import *
import operator
from src.data_manager import DataManager

class BayesClassifier:
    """
    * Not working when one class contains much more samples.
    """

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

        task_classes = list(classes.keys())
        ordered_data = DataManager.order_data(inputs, target, task_classes)

        # categorize continous data
        for task_class in task_classes:
            for category in categories:
                cat_index = categories.index(category)
                # contious data
                if categorical_mask[cat_index] == False:
                   min_val = min(ordered_data[task_class][:, cat_index])
                   max_val = max(ordered_data[task_class][:, cat_index])
                   feature_prob = BayesClassifier.compute_feature_prop(ordered_data[task_class][:, cat_index],8, min_val, max_val)
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
                    # compute labeled category probs
                    cat_values = list(class_cate_vals.values())
                    sum_cat_values = sum(cat_values)
                    for cate_item in classes[class_item][category]:
                        classes[class_item][category][cate_item] = classes[class_item][category][cate_item] / sum_cat_values
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


    def test_classify(self, classes: dict, input: ndarray, target: ndarray, class_props: dict, class_features_diffs: dict) -> list:
        """
        Classifies task
        :return: list with tuples that contains estimated values and target
        """
        results = []
        for index, test_input in enumerate(input):
            est_class, est_class_prob = self.classify_item(test_input, classes, class_props, class_features_diffs)
            results.append((target[index], (est_class, est_class_prob)))
        return results


    def classify_item(self, item: dict, classes: dict, class_props: dict, class_features_diffs: dict) -> float:
        """
        Computes the final result of selecting an item and probability of selecting it.
        :param item: Item to classify
        :param classes:
        :return: estimated class
        """
        est_props = {}
        for class_item in classes:

            # for this class get item probs

            prop_sum = 1
            for category in classes[class_item]:
                item_cat_value = item[category]
                if type(item_cat_value) == float:
                        category_props = classes[class_item][category]
                        p_cat = self.compute_item_feature_fit(item_cat_value, category_props)
                        class_feat_diff = class_features_diffs[category]
                        # prop_sum +=  p_cat * p_class / len(list(classes.keys()))
                        prop_sum *= p_cat
                else:
                    if item[category] not in classes[class_item][category]:
                        print("category: {} not in class: {}".format(category, class_item))
                        continue
                    prop_sum *= classes[class_item][category][item[category]]
            item_length = len(item)
            est_props[class_item] = prop_sum * class_props[class_item]

        # sum props
        # prop of selecting the right class

        result = max(est_props.items(), key=operator.itemgetter(1))[0]
        result_prob = est_props[result] / sum(list(est_props.values()))
        return result, result_prob

    def compute_class_features_diffs(self, data: ndarray, categories: list, categorical_mask: list) -> dict:
        """
        Works only for float categories.
        Computes differences between continuous data features.
        Larger value means that differences between classes features are larger.
        :param data: ndarray with all data features
        :param categories: data categories names list
        :param categorical_mask: tells weather feature is float or string
        :return: dict with classes features differences.
        """
        class_features_diffs = {}
        for cat_index in range(len(categories)):
            if not categorical_mask[cat_index]:
                class_feature_diff = BayesClassifier.class_feature_diff(data[0][:, cat_index], data[1][:, cat_index], 8)
                class_features_diffs[categories[cat_index]] = class_feature_diff
        return class_features_diffs


    @staticmethod
    def compute_feature_prop(data: ndarray, sections: int, min_val, max_val):
        """
        :param data:
        :param sections:
        :param min_val:
        :param max_val:
        :return:
        """
        if data.ndim != 1:
            raise ValueError("Data is not a vector.")
        feature_prop = MathOper.get_prop_data(data, sections, min_val, max_val)
        return feature_prop

    @staticmethod
    def class_feature_diff(feature_class_1: ndarray, feature_class_2: ndarray, sections: int):
        """
        Computes props diff between features
        :return:
        """
        if feature_class_1.ndim != 1 or feature_class_2.ndim != 1:
            raise ValueError("Feature data ndarrays are not vectors.")
        # get larger list set margin
        test_list = append(feature_class_1, feature_class_2)
        feat_min = min(test_list)
        feat_max = max(test_list)

        feature_props_1 = BayesClassifier.compute_feature_prop(feature_class_1, sections, feat_min, feat_max)
        feature_props_2 = BayesClassifier.compute_feature_prop(feature_class_2, sections, feat_min, feat_max)
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



