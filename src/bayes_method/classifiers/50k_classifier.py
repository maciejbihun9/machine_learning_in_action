
from src.data_manager import DataManager
from numpy import *
from src.math_oper import MathOper
from src.credibility import Credibility
from src.bayes_method.bayes_classification.bayes_classifier import BayesClassifier

# task init
url = '../../resources/50k.txt'
data = DataManager.load_data(url, False, True, ', ')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 25000
test_N = 10000
task_classes = [0, 1]

categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]

# data preparing
train_inputs = data[0:N, 0:14]

test_inputs = data[test_N:N]

targets = data[0:N, 14]

targets = array([0 if '<=50' in y else 1 for y in targets])

train_targets = targets[0:N]

test_targets = targets[test_N:N]

ordered_data = DataManager.order_data(train_inputs, train_targets, task_classes)

ordered_test_data = DataManager.order_data(test_inputs, test_targets, task_classes)

class_props = MathOper.get_classes_prop(train_targets, task_classes)

m, n = shape(train_inputs)

# init classifier
bayes_classifier = BayesClassifier()

classes = bayes_classifier.init_classes(train_inputs, train_targets, categories, categorical_mask, task_classes, True)

class_features_diffs = bayes_classifier.compute_class_features_diffs(ordered_test_data, categories, categorical_mask)

test_inputs = bayes_classifier.prepare_test_items(test_inputs, categories)
results = bayes_classifier.test_classify(classes, test_inputs, test_targets, class_props, class_features_diffs)

# compute full credibility
credibility = Credibility(results, task_classes, class_props)
precision = credibility.get_precision()
print("precision: {}".format(precision))
