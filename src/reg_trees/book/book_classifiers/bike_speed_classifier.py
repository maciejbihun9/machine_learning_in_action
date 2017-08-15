from numpy import *
from src.reg_trees.book.tree_reg_classifier import *
import matplotlib.pyplot as plt


trainMat = mat(loadDataSet('../../../../resources/trees/bikeSpeedVsIq_train.txt'))
# plt.plot(array(trainMat)[:, 0], array(trainMat)[:, 1], 'ro')
# plt.show()
testMat = mat(loadDataSet('../../../../resources/trees/bikeSpeedVsIq_test.txt'))
"""
myTree = createTree(trainMat, ops=(1,20))
print(myTree)
yHat = createForeCast(myTree, testMat[:,0])
plt.plot(array(testMat[:, 0]), array(yHat[:, 0]), 'ro')
plt.show()
print("corcoeff : {}".format(corrcoef(yHat[:, 0], testMat[:,0], rowvar=0)[0,1]))
"""

# regression tree
yHat = [0.0] * len(testMat)
ws, X, Y = linearSolve(trainMat)
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i,0] * ws[1,0] + ws[0,0]
plt.plot(testMat[:, 0], yHat, 'ro')
plt.show()
cor = corrcoef(yHat, testMat[:,1], rowvar = 0)[0,1]
print("Correlation : {}".format(cor))