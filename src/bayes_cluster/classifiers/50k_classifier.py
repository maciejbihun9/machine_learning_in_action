from src.data_manager import DataManager
from numpy import *
from src.bayes_cluster.bayes_cluster_classifier import BayesClusterClassifier
from src.bayes_method.bayes_classification.bayes_classifier import BayesClassifier
from src.class_manager import ClassManager

# task init
url = '../../../resources/50k.txt'
data = DataManager.load_data(url, False, True, ', ')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 25000
test_N = 10000
task_classes = [0, 1]

categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# create category data frame

# this should be a job for data manager

# categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]

# we need this type of categorial mask. We do not know if numerical values is always a numerical value
cate_mask = {'age' : False, 'workclass' : True, 'fnlwgt': False, 'education' : True, 'education-num' : False, 'marital-status': True,
             'occupation' : True, 'relationship' : True, 'race' : True, 'sex' : True, 'capital-gain'  : False, 'capital-loss' : False, 'hours-per-week' : False, 'native-country': True}

# data preparing
train_inputs = data[0:N, 0:14]

test_inputs = data[test_N:N]

targets = data[0:N, 14]

targets = array([0 if '<=50' in y else 1 for y in targets])

train_targets = targets[0:N]

train_inputs_dict = DataManager.create_data_frame(train_inputs, categories)

test_targets = targets[test_N:N]

ordered_data = DataManager.order_data(train_inputs, train_targets, task_classes)

ordered_test_data = DataManager.order_data(test_inputs, test_targets, task_classes)

# init classes dict only with labeled data probs
classes = {}
classes = ClassManager.add_labeled_skeleton(classes, task_classes, cate_mask)

classes = ClassManager.init_classes_skeleton_with_labeled_data(classes, train_inputs_dict, train_targets, cate_mask)
classes = ClassManager.replace_labeled_data_with_probs(classes, cate_mask)
# ClassManager.replace_labeled_data_with_probs(classes, )
print(classes)

# predictions = BayesClusterClassifier.predict()