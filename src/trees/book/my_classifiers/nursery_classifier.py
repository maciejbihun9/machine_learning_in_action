from src.trees.book.tree_classifier import *
from src.trees.book.tree_ploter import *
from numpy import *

fr = open("../../../../resources/nursery.txt")
# parsed dataset
data_items = [inst.strip().split(',') for inst in fr.readlines()]
# lenses classes
data_items = array(data_items)

items = data_items[:, 0:8]
class_items = data_items[:, 8].tolist()
class_items = [class_items]
class_items = array(class_items)
class_items = array(class_items).T

train_items = items[0:10000, :]
train_classes = class_items[0:10000]

test_min = 10000
test_max = 12960
test_items_margin = test_max - test_min
test_items = items[test_min: test_max, :]
test_classes = class_items[test_min: test_max]

train_data = append(items, class_items, axis=1)
train_data = train_data.tolist()

"""
parents        usual, pretentious, great_pret
   has_nurs       proper, less_proper, improper, critical, very_crit
   form           complete, completed, incomplete, foster
   children       1, 2, 3, more
   housing        convenient, less_conv, critical
   finance        convenient, inconv
   social         non-prob, slightly_prob, problematic
   health
"""

classes = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']
features = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
tree = createTree(train_data, features[:])
correct_answers = 0
for index, test_item in enumerate(test_items):
    result = classify(tree, features, test_item.tolist())
    test_class_result = test_classes[index]
    if result == test_classes[index]:
        correct_answers += 1

print("Classification results : {}".format(correct_answers / test_items_margin))
createPlot(tree)