from numpy import *

from src.bayes_cluster.bayes_cluster_classifier import BayesClusterClassifier
from src.class_manager import ClassManager
from src.credibility import Credibility
from src.data_manager import DataManager
from src.math_oper import MathOper

# task init
url = '../../../resources/50k.txt'
data = DataManager.load_data(url, False, True, ', ')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 25000
task_classes = [0, 1]

categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

cate_mask = {'age' : False, 'workclass' : True, 'fnlwgt': False, 'education' : True, 'education-num' : False, 'marital-status': True,
             'occupation' : True, 'relationship' : True, 'race' : True, 'sex' : True, 'capital-gain'  : False, 'capital-loss' : False, 'hours-per-week' : False, 'native-country': True}

# data preparing
train_inputs = data[0:20000, 0:14]

test_inputs = data[20001:N, 0:14]

targets = data[0:N, 14]

targets = array([0 if '<=50' in y else 1 for y in targets])

train_targets = targets[0:20000]

train_inputs_dict = DataManager.create_data_frame(train_inputs, categories)

test_inputs_dict = DataManager.create_data_frame(test_inputs, categories)

test_targets = targets[20001:N]

class_props = MathOper.get_classes_prop(train_targets, task_classes)

# Visual.plot_prop_dist(data[:, 2], 8)

classes = {}
classes = ClassManager.add_labeled_skeleton(classes, task_classes, cate_mask)
classes = ClassManager.add_numerical_skeleton(classes, task_classes, cate_mask)

classes = ClassManager.init_classes_skeleton_with_labeled_data(classes, train_inputs_dict, train_targets, cate_mask)
classes = ClassManager.init_classes_skeleton_with_numerical_data(classes, train_inputs_dict, train_targets, cate_mask)

classes = ClassManager.replace_numerical_data_with_means(classes, cate_mask)
classes = ClassManager.replace_labeled_data_with_probs(classes, cate_mask)

results = BayesClusterClassifier.predict(classes, test_inputs_dict, test_targets, cate_mask)

credibility = Credibility(results, task_classes, class_props)
predictions = credibility.get_precision()

print(predictions)

