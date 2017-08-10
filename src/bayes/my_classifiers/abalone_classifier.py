
from numpy import *

from src.bayes.bayes_classifier import BayesClassifier
from src.credibility import Credibility
from src.data_manager import DataManager
from src.math_oper import MathOper

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

# categorical_mask = [False, False, False, False, False, False, False, False]

cate_mask = {'Length' : False, 'Diameter' : False, 'Height' : False, 'Whole weight' : False,
             'Shucked weight' : False, 'Viscera weight' : False, 'Shell weight' : False, 'Rings' : False}
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

ordered_test_data = DataManager.order_data(test_inputs, test_target, task_classes)

class_props = MathOper.get_classes_prop(train_target, task_classes)

m, n = shape(train_inputs)

# init classifier
bayes_classifier = BayesClassifier()

classes = bayes_classifier.init_classes(train_inputs, train_target, cate_mask, task_classes, True)

test_inputs = bayes_classifier.prepare_test_items(test_inputs, categories)
results = bayes_classifier.test_classify(classes, test_inputs, test_target, class_props)

# compute full credibility
credibility = Credibility(results, task_classes, class_props)
precision = credibility.get_precision()
print("precision: {}".format(precision))
