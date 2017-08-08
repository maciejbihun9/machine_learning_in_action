
from src.data_manager import DataManager
from numpy import *
from src.math_oper import MathOper
from src.credibility import Credibility
from src.bayes_method.bayes_classification.bayes_classifier import BayesClassifier
from src.visual import Visual

# task init
url = '../../../resources/abalone.txt'
data = DataManager.load_data(url, False, True, ',')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 4117
test_N = 4117

not_parsed_classes = ["M", "F", "I"]

task_classes = [0, 1, 2]

categories = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

categorical_mask = [False, False, False, False, False, False, False, False]

# categories = ['Length', 'Diameter']

# categorical_mask = [False, False]


# data preparing
train_inputs = data[0:N, 1:9]

test_inputs = data[0:test_N, 1:9]

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

# compute full credibility
credibility = Credibility(results, task_classes, class_props)
precision = credibility.get_precision()
print("precision: {}".format(precision))
