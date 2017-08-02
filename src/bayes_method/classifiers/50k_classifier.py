
from src.data_manager import DataManager
from numpy import *
from src.visual import Visual
from src.math_oper import MathOper
from src.normalizer import Normalizer
from src.norm_type import NormType
from src.bayes_method.bayes_classification.bayes_classifier import BayesClassifier
import operator
import matplotlib.pyplot as plt


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
test_N = 50

categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]

inputs = data[0:N, 0:14]
test_inputs = data[0:test_N]

target = data[0:N, 14]
target = array([0 if '<=50' in y else 1 for y in target])
test_target = target[0:N]

item_occurances = DataManager.item_occurances(target)

task_classes = [0, 1]
ordered_data = DataManager.order_data(inputs, target, task_classes)
"""
for categorized_data_item in ordered_data:
    Visual.plot_hist(categorized_data_item[:, 0])
"""
"""
props = MathOper.get_prop_data(inputs[:,12], 50)
plt.plot(props[:,0], props[:,1])
plt.show()"""

props = Visual.plot_prop_dist(ordered_data[0][:, 12], 100)

class_props = MathOper.get_classes_prop(target, task_classes)

# count classes occurances
class_counts = DataManager.item_occurances(target)

# create a list with dicts that stores class params
m, n = shape(inputs)

"""
* Classifier data end
"""
# ----------------------------------------------------------------------------------------------------------------------
"""
* Mine the data
"""
# init classes with data
bayes_classifier = BayesClassifier()
classes = bayes_classifier.init_classes(inputs, target, categories, categorical_mask, task_classes)

test_inputs = bayes_classifier.prepare_test_items(test_inputs, categories)
results = bayes_classifier.test_classify(classes, test_inputs, test_target, class_props)
print(results)
"""
* Mine data end
"""

