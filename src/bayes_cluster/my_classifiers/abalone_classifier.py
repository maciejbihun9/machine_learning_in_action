
from src.data_manager import DataManager
from numpy import *
from src.math_oper import MathOper
from src.credibility import Credibility
from src.class_manager import ClassManager
from src.bayes_cluster.bayes_cluster_classifier import BayesClusterClassifier

# task init
url = '../../../resources/abalone.txt'
data = DataManager.load_data(url, False, True, ',')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 4117

not_parsed_classes = ["M", "F", "I"]

task_classes = [0, 1, 2]

categories = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

categorical_mask = [False, False, False, False, False, False, False, False]

cate_mask = {'Length' : False, 'Diameter' : False, 'Height' : False, 'Whole weight' : False,
             'Shucked weight' : False, 'Viscera weight' : False, 'Shell weight' : False, 'Rings' : False}

# data preparing
train_inputs = data[0:3000, 1:9]

test_inputs = data[3001:N, 1:9]

target = data[:, 0]

target = DataManager.assign_classes(target)

train_targets = target[0:3000]

test_targets = target[3001:N]

train_inputs_dict = DataManager.create_data_frame(train_inputs, categories)

test_inputs_dict = DataManager.create_data_frame(test_inputs, categories)

ordered_data = DataManager.order_data(train_inputs, train_targets, task_classes)

class_props = MathOper.get_classes_prop(train_targets, task_classes)

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

