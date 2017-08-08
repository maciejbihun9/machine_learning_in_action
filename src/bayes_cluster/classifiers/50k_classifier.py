from src.data_manager import DataManager
from numpy import *
from src.bayes_cluster.bayes_cluster_classifier import BayesClusterClassifier

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

# get init classes
# get classes without numerical values
# i should create seperate methods for classes init with labels and numerical valuea

predictions = BayesClusterClassifier.predict()