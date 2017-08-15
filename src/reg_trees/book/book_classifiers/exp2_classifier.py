
from src.reg_trees.book.tree_reg_classifier import *
from numpy import *
import matplotlib.pyplot as plt

myDat = loadDataSet('../../../../resources/trees/exp2.txt')
plt.plot(array(myDat)[:, 0], array(myDat)[:, 1], 'ro')
plt.show()
myMat = mat(myDat)
tree = createTree(myMat, modelLeaf, modelErr, ops = (1, 10))
myDatTest = loadDataSet('../../../../resources/trees/ex2test.txt')
myDatTest = mat(myDatTest)
tree = prune(tree, myDatTest)
print("tree : {}".format(tree))