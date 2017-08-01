
from numpy import *

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


def test_classify(classes: dict, test_inputs: ndarray, test_targets: ndarray):
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
    for item in item_values:
        if item_value > item:
            return item_values[item]

def classify_item(item: dict, classes: dict, class_props: list) -> float:
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
                    prop_sum += get_item_prop(item_cat_value, category_prop)
                except:
                    print("Error")
                prop_sum += log(class_props[class_item])

            else:
                prop_sum += classes[class_item][category][item[category]] + log(class_props[class_item])
        est_props[class_item] = prop_sum
    result = max(est_props.items(), key=operator.itemgetter(1))[0]
    return result