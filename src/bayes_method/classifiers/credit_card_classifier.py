
from src.data_manager import DataManager
from numpy import *
from src.math_oper import MathOper
from src.credibility import Credibility
from src.bayes_method.bayes_classification.bayes_classifier import BayesClassifier
from src.visual import Visual

# task init
url = '../../../resources/credit_card.txt'
data = DataManager.load_data(url, False, True, ',')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 500
test_N = 600
task_classes = [0, 1]

categories = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']

categorical_mask = [True, False, False, True, True, True, True, False, True, True, False, True, True, False, False]

# data preparing
inputs = data[0:N, 0:15]

test_inputs = data[N:test_N, 0:15]

target = data[0:N, 15]

target = array([0 if '-' in y else 1 for y in target])

test_target = data[N:test_N, 15]

test_target = array([0 if '-' in y else 1 for y in test_target])

ordered_data = DataManager.order_data(inputs, target, task_classes)

# visual ordered data
# Visual.plot_prop_dist(ordered_data[0][:, 2], 40)

ordered_test_data = DataManager.order_data(test_inputs, test_target, task_classes)

class_props = MathOper.get_classes_prop(target, task_classes)

m, n = shape(inputs)

# init classifier
bayes_classifier = BayesClassifier()

classes = bayes_classifier.init_classes(inputs, target, categories, categorical_mask, task_classes)

class_features_diffs = bayes_classifier.compute_class_features_diffs(ordered_test_data, categories, categorical_mask)

test_inputs = bayes_classifier.prepare_test_items(test_inputs, categories)
results = bayes_classifier.test_classify(classes, test_inputs, test_target, class_props, class_features_diffs)

# compute full credibility
credibility = Credibility(results)
predictions = credibility.get_predictions()
"""
accuracy = credibility.get_accuracy()
precision = credibility.get_precision()
sensitivity = credibility.get_sensitivity()
specificity = credibility.get_specificity()
f_score = credibility.get_f_score()
"""
print(results)
