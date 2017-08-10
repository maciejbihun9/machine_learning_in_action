from sklearn.naive_bayes import GaussianNB
from src.data_manager import DataManager
from numpy import *
from src.credibility import Credibility
from src.math_oper import MathOper

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

train_inputs = data[0:N, 1:9]

test_inputs = data[0:test_N, 1:9]

target = data[:, 0]

target = DataManager.assign_classes(target)

train_target = target[0:N]

test_target = target[0:test_N]

class_props = MathOper.get_classes_prop(train_target, task_classes)

gnb = GaussianNB()
y_pred = gnb.fit(train_inputs, train_target)
results = []
for test_inp_index, test_input in enumerate(test_inputs):
    test_input = test_input.reshape(-1, 1)
    result = y_pred.predict(test_input)
    results.append((test_target[test_inp_index] ,result))

credibility = Credibility(results, task_classes, class_props)
predictions = credibility.get_precision()
print("results: {}".format(results))
