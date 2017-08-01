
from src.data_manager import DataManager
from numpy import *
from src.visual import Visual
from src.math_oper import MathOper
from src.normalizer import Normalizer
from src.norm_type import NormType
import operator


"""
* Classifier data
"""
url = '../../resources/50k.txt'
data = DataManager.load_data(url, False, True, ', ')
data = array(data, dtype='object')

# filter
no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 10000
test_N = 200

categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]

inputs = data[0:N, 0:14]
test_inputs = data[0:test_N]

# Visual.plot_hist(test)
# inputs = Normalizer.normalize(inputs, NormType.stand_norm, [0, 2, 4, 10, 11, 12])

# filter data

target = data[0:N, 14]
target = array([0 if '<=50' in y else 1 for y in target])
test_target = data[0:test_N]

task_classes = [0, 1]
class_props = MathOper.get_classes_prop(target, task_classes)

# count classes occurances
class_counts = DataManager.item_occurances(target)

# create a list with dicts that stores class params
m, n = shape(inputs)

"""
* Classifier data end
"""

# compute classes

# dict for class data
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
print("res")

def prepare_test_items(items: ndarray, categories: list) -> list:
    """

    :param items:
    :return: list with dicts
    """
    items_dicts = []
    for item in items:
        item_dict = dict(zip(categories, item))
        items_dicts.append(item_dict)
    return items_dicts


def test_classify(classes: dict, test_inputs: ndarray, test_targets: ndarray, class_props: dict):
    # for each test item
    right_answeres = 0
    for index, test_input in enumerate(test_inputs):
        # classify item
        est_class = classify_item(test_input, classes, class_props)
        if est_class == test_targets[index]:
            right_answeres += 1
    correctness = right_answeres / len(test_inputs)
    return correctness

def get_item_prop(item_value: float, item_values: list):
    try:
        for item in item_values:
            if item_value > item:
                return item_values[item]
    except:
        print("Error")

def classify_item(item: dict, classes: dict, class_props: dict) -> float:
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
                # get first larger value
                try:
                    category_prop = classes[class_item][category]
                    print("Category: {}".format(category))
                    prop_sum += get_item_prop(item_cat_value, category_prop) + log(class_props[class_item])
                except:
                    print("Error")

            else:
                prop_sum += classes[class_item][category][item[category]] + log(class_props[class_item])
        est_props[class_item] = prop_sum
    result = max(est_props.items(), key=operator.itemgetter(1))[0]
    return result

test_inputs = prepare_test_items(test_inputs, categories)
results = test_classify(classes, test_inputs, test_target, class_props)
