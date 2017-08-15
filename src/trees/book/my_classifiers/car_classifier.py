from src.trees.book.tree_classifier import *
from src.trees.book.tree_ploter import *
from numpy import *

fr = open("../../../../resources/car.txt")
# parsed dataset
data_items = [inst.strip().split(',') for inst in fr.readlines()]
# lenses classes
data_items = array(data_items)

items = data_items[:, 0:6]
class_items = data_items[:, 6].tolist()
class_items = [class_items]
class_items = array(class_items)
class_items = array(class_items).T

train_items = items[0:1200, :]
train_classes = class_items[0:1200]

test_min = 1201
test_max = 1728
test_items_margin = test_max - test_min
test_items = items[test_min: test_max, :]
test_classes = class_items[test_min: test_max]

train_data = append(items, class_items, axis=1)
train_data = train_data.tolist()

classes = ['unacc', 'acc', 'good', 'vgood']
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
tree = createTree(train_data, features[:])
correct_answers = 0
for index, test_item in enumerate(test_items):
    result = classify(tree, features, test_item.tolist())
    test_class_result = test_classes[index]
    if result == test_classes[index]:
        correct_answers += 1

print("Classification results : {}".format(correct_answers / test_items_margin))
# createPlot(tree)