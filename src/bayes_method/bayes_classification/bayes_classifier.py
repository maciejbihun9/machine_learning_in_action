
from src.math_oper import MathOper
from numpy import *

def init_classes(inputs: ndarray, target: ndarray, categories: list, categorical_mask: list, task_classes: list) -> dict:
    """
    :param inputs: ndarray with data inputs
    :param target: ndarray with data targets
    :param categories: list with all inputs categories
    :param categorical_mask: list with categories masks
    :param task_classes: Problem available classes
    :return: initialized classes dict
    """
    classes = {}
    # create a classes dict
    # for each class create categories
    for task_class in task_classes:
        classes[task_class] = {}
        for category in categories:
            cat_index = categories.index(category)
            # categorical mask
            if categorical_mask[cat_index] == True:
                # we can not init it with possible categoris because we do not know what they are
                classes[task_class][category] = {}
            else:
                classes[task_class][category] = []


                # for each item in inputs list
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
            # continuous category
            else:
                # classes[target[index]][categories[cat_index]] += item[cat_index]
                classes[target[index]][categories[cat_index]].append(item[cat_index])

    # for each class
    for class_item in classes:
        # we have continues value
        for index, category in enumerate(classes[class_item]):
            # continous value
            if categorical_mask[index] == False:
                class_cate_vals = classes[class_item][category]
                # compute points and props
                cate_points, cate_props = MathOper.get_data_prop(class_cate_vals, 100)
                classes[class_item][category] = dict(zip(cate_props, cate_points))
            # label value
            else:
                class_cate_vals = classes[class_item][category]
                # sum category value
                class_cate_sum_val = 0
                for cate_value in classes[class_item][category]:
                    class_cate_sum_val += classes[class_item][category][cate_value]
                for cate_item in classes[class_item][category]:
                    cate_item_value = classes[class_item][category][cate_item]
                    classes[class_item][category][cate_item] = cate_item_value / class_cate_sum_val

    return classes
