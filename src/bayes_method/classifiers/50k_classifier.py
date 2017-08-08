from src.norm_type import NormType
from src.normalizer import Normalizer
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

N = 2500
test_N = 1000
task_classes = [0, 1]

categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]


# data preparing
inputs = data[0:N, 0:14]

# inputs = Normalizer.normalize(inputs, NormType.data_norm, [0,2,4,10,11,12])

test_inputs = data[0:test_N]

target = data[0:N, 14]

target = array([0 if '<=50' in y else 1 for y in target])

test_target = target[0:N]

ordered_data = DataManager.order_data(inputs, target, task_classes)

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
accuracy = credibility.get_accuracy()
precision = credibility.get_precision()
sensitivity = credibility.get_sensitivity()
specificity = credibility.get_specificity()
f_score = credibility.get_f_score()
print(results)
