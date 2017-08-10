
from numpy import *

from src.bayes.bayes_classifier import BayesClassifier
from src.credibility import Credibility
from src.data_manager import DataManager
from src.math_oper import MathOper
from src.similarity import Similarity

# task init
url = '../../../resources/50k.txt'
data = DataManager.load_data(url, False, True, ', ')
data = array(data, dtype='object')

no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

N = 25000
task_classes = [0, 1]

categories = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]

cate_mask = {'age' : False, 'workclass' : True, 'fnlwgt': False, 'education' : True, 'education-num' : False, 'marital-status': True,
             'occupation' : True, 'relationship' : True, 'race' : True, 'sex' : True, 'capital-gain'  : False, 'capital-loss' : False, 'hours-per-week' : False, 'native-country': True}

# data preparing
train_inputs = data[0:20000, 0:14]

test_inputs = data[20001:N]

targets = data[0:N, 14]

targets = array([0 if '<=50' in y else 1 for y in targets])

train_targets = targets[0:20000]

test_targets = targets[20001:N]

ordered_data = DataManager.order_data(train_inputs, train_targets, task_classes)

ordered_test_data = DataManager.order_data(test_inputs, test_targets, task_classes)

# checking data similarity
similarity = Similarity()
jaccard_age_similarity = similarity.jaccard_similarity(ordered_data[0][:, 0], ordered_data[1][:, 0])
jaccard_fnlwgt_similarity = similarity.jaccard_similarity(ordered_data[0][:, 2], ordered_data[1][:, 2])
jaccard_education_num_similarity = similarity.jaccard_similarity(ordered_data[0][:, 4], ordered_data[1][:, 4])

manhattan_distance = similarity.manhattan_distance(ordered_data[0][0], ordered_data[1][0])
minkowski_distance = similarity.minkowski_distance(ordered_data[0][0], ordered_data[1][0])


class_props = MathOper.get_classes_prop(train_targets, task_classes)

m, n = shape(train_inputs)

# init classifier
bayes_classifier = BayesClassifier()

classes = bayes_classifier.init_classes(train_inputs, train_targets, cate_mask, task_classes, True)

test_inputs = bayes_classifier.prepare_test_items(test_inputs, categories)
results = bayes_classifier.test_classify(classes, test_inputs, test_targets, class_props)

# compute full credibility
credibility = Credibility(results, task_classes, class_props)
precision = credibility.get_precision()
print("precision: {}".format(precision))
