
from src.data_manager import DataManager
from src.bayes_method.bayes_classification.bayes_classifier import BayesClassifier
from numpy import *

class ClassManager:


    @staticmethod
    def add_labeled_skeleton(classes: dict, task_classes: list, categories: dict) -> dict:
        """
        Initializes classes dict with labeled category skeleton.
        :param classes: dict
        :param task_classes: target classes list.
        :param categories: task categories names list.
        :param categorical_mask: list with info about type of the data category.
        :return: dict with skeleton build from label category data.
        """
        for task_class in task_classes:
            if task_class not in classes:
                classes[task_class] = {}
            for category in categories:
                if categories[category]:
                    classes[task_class][category] = {}
        return classes

    @staticmethod
    def add_numerical_skeleton(classes: dict, task_classes: list, categories: dict) -> dict:
        """
        Initializes classes dict with numerical category skeleton.
        :param classes: dict
        :param task_classes: target classes list.
        :param categories: task categories names list.
        :param categorical_mask: list with info about type of the data category.
        :return: dict with skeleton build from numerical category data.
        """
        for task_class in task_classes:
            if task_class not in classes:
                classes[task_class] = {}
            for category in categories:
                if not categories[category]:
                    classes[task_class][category] = []
        return classes

    @staticmethod
    def init_classes_skeleton_with_labeled_data(classes: dict, inputs: dict, target: ndarray, categories: dict):
        """
        Initializes classes skeleton with labeled data.
        :param classes:
        :param inputs:
        :param target:
        :param categories:
        :param categorical_mask:
        :return:
        """

        for class_item in classes:
            for category in inputs:
                # labeled category
                if categories[category]:
                    for index, target_item in enumerate(target):
                        if target_item == class_item:
                            if inputs[category][index] in classes[class_item][category]:
                                classes[class_item][category][inputs[category][index]] += 1
                            else:
                                classes[class_item][category][inputs[category][index]] = 0
        return classes

    @staticmethod
    def init_classes_skeleton_with_numerical_data(classes: dict, inputs: dict, target: ndarray, categories: list):
        for class_item in classes: # 2
            for category in inputs: # 'age' - []
                # labeled category
                if not categories[category]: # if numerical
                    for index, target_item in enumerate(target):
                        if target_item == class_item: # if target is for that class
                            classes[class_item][category].append(inputs[category][index])
        return classes

    @staticmethod
    def replace_numerical_data_with_means(classes: dict, categories: dict):
        """
        Replaces labeled data with probabilities of selecting an item.
        :param classes:
        :param categorical_mask:
        :return:
        """
        for class_item in classes:
            for index, category in enumerate(classes[class_item]): #
                if not categories[category]: # if not True
                    class_cate_vals = classes[class_item][category] #
                    mean_value = mean(class_cate_vals)
                    classes[class_item][category] = mean_value
        return classes

    @staticmethod
    def replace_labeled_data_with_probs(classes: dict, categories: dict):
        """
        Replaces labeled data with probabilities of selecting an item.
        :param classes:
        :param categorical_mask:
        :return:
        """
        for class_item in classes:
            for index, category in enumerate(classes[class_item]):
                if categories[category]:
                    class_cate_vals = classes[class_item][category]
                    cat_values = list(class_cate_vals.values())
                    sum_cat_values = sum(cat_values)
                    for cate_item in classes[class_item][category]:
                        classes[class_item][category][cate_item] = round(classes[class_item][category][cate_item] / sum_cat_values, 4)
        return classes

    @staticmethod
    def categorize_data(classes: dict, inputs: ndarray, target: ndarray, categories: list,
                         categorical_mask: list, numerical_init: bool) -> dict:
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

        # categorize numerical data
        if numerical_init:
            for task_class in task_classes:
                for category in categories:
                    cat_index = categories.index(category)
                    # contious data
                    if not categorical_mask[cat_index]:
                        min_val = min(ordered_data[task_class][:, cat_index])
                        max_val = max(ordered_data[task_class][:, cat_index])
                        feature_prob = BayesClassifier.compute_feature_prop(ordered_data[task_class][:, cat_index], 8,
                                                                            min_val, max_val)
                        classes[task_class][category] = feature_prob

        # categorize labeled data
        for index, item in enumerate(inputs):
            # each item is a row
            for category in categories:
                # for categorical item
                cat_index = categories.index(category)
                if categorical_mask[cat_index]:
                    if item[cat_index] in classes[target[index]][categories[cat_index]]:
                        classes[target[index]][categories[cat_index]][item[cat_index]] += 1
                    else:
                        classes[target[index]][categories[cat_index]][item[cat_index]] = 0
        return classes

    def init_classes(self, inputs: ndarray, target: ndarray, categories: list, categorical_mask: list,
                     task_classes: list, numerical_init: bool) -> dict:
        """
        :param inputs: ndarray with data inputs
        :param target: ndarray with data targets
        :param categories: list with all inputs categories
        :param categorical_mask: list with categories masks
        :param task_classes: Problem available classes
        :return: dict with categorized data and adjusted probability.
        """
        classes = {}
        classes = self._add_labeled_skeleton(classes, task_classes, categories, categorical_mask)
        classes = self._add_numerical_skeleton(classes, task_classes, categories, categorical_mask)
        # classes = self._create_classes_skeleton(task_classes, categories, categorical_mask)
        classes = self._categorize_data(classes, inputs, target, categories, categorical_mask, numerical_init)

        # compute data probability
        for class_item in classes:
            for index, category in enumerate(classes[class_item]):
                if categorical_mask[index]:
                    class_cate_vals = classes[class_item][category]
                    # compute labeled category probs
                    cat_values = list(class_cate_vals.values())
                    sum_cat_values = sum(cat_values)
                    for cate_item in classes[class_item][category]:
                        classes[class_item][category][cate_item] = classes[class_item][category][
                                                                       cate_item] / sum_cat_values
        return classes