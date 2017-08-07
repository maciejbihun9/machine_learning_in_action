
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from src.data_manager import DataManager
from numpy import *

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
inputs = data[0:N, 0:14]

test_inputs = data[0:test_N]

target = data[0:N, 14]

target = array([0 if '<=50' in y else 1 for y in target])

test_target = target[0:N]


gnb = GaussianNB()
y_pred = gnb.fit(inputs, target).predict(test_target)
print("results")
# print("Number of mislabeled points out of a total %d points : %d"
  #    % (iris.data.shape[0],(iris.target != y_pred).sum()))