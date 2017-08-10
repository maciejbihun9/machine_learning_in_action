
from numpy import *

from src.bayes.bayes_classifier import BayesClassifier
from src.credibility import Credibility
from src.data_manager import DataManager
from src.math_oper import MathOper

# task init
url = '../../../resources/mashrooms.txt'
data = DataManager.load_data(url, False, True, ',')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 8124
test_N = 8124

not_parsed_classes = ['p', 'e']

task_classes = [0, 1]

categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','i', 'j', 'k', 'l', 'ł', 'm', 'n', 'o','p', 'r', 's', 't', 'w', 'z']

categorical_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

# data preparing
train_inputs = data[0:N, 1:23]

test_inputs = data[0:test_N, 1:23]

target = data[:, 0]

target = DataManager.assign_classes(target)

train_target = target[0:N]

test_target = target[0:test_N]

ordered_data = DataManager.order_data(train_inputs, train_target, task_classes)

# visual ordered data
# Visual.plot_prop_dist(ordered_data[0][:, 1], 40)
# Visual.plot_prop_dist(ordered_data[1][:, 1], 40)
# Visual.plot_prop_dist(ordered_data[2][:, 1], 40)



ordered_test_data = DataManager.order_data(test_inputs, test_target, task_classes)

class_props = MathOper.get_classes_prop(train_target, task_classes)

m, n = shape(train_inputs)

# init classifier
bayes_classifier = BayesClassifier()

classes = bayes_classifier.init_classes(train_inputs, train_target, categories, categorical_mask, task_classes)

class_features_diffs = bayes_classifier.compute_class_features_diffs(ordered_test_data, categories, categorical_mask)

test_inputs = bayes_classifier.prepare_test_items(test_inputs, categories)
results = bayes_classifier.test_classify(classes, test_inputs, test_target, class_props, class_features_diffs)

credibility = Credibility(results, task_classes, class_props)
precision = credibility.get_precision()
print("precision: {}".format(precision))
