from numpy import *

from src.bayes.bayes_classifier import BayesClassifier
from src.credibility import Credibility
from src.data_manager import DataManager
from src.math_oper import MathOper

# task init
url = '../../../resources/balance.txt'
data = DataManager.load_data(url, False, True, ',')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 625
test_N = 625

not_parsed_classes = ["L", "B", "R"]

task_classes = [0, 1, 2]

categories = ['a', 'b', 'c', 'd']

# categorical_mask = [False, False, False, False, False, False, False, False]

cate_mask = {'Length' : False, 'Diameter' : False, 'Height' : False, 'Whole weight' : False,
             'Shucked weight' : False, 'Viscera weight' : False, 'Shell weight' : False, 'Rings' : False}
# categories = ['Length', 'Diameter']

# categorical_mask = [False, False]


# data preparing
train_inputs = data[0:500, 1:9]

test_inputs = data[501:625, 1:9]

target = data[:, 0]

target = DataManager.assign_classes(target)

train_target = target[0:500]

test_target = target[501:625]

ordered_data = DataManager.order_data(train_inputs, train_target, task_classes)

ordered_test_data = DataManager.order_data(test_inputs, test_target, task_classes)

class_props = MathOper.get_classes_prop(train_target, task_classes)