
from src.tkinker import Tkinker
from numpy import *
from src.reg_trees.book.tree_reg_classifier import *

dataset = mat(loadDataSet('../resources/trees/bikeSpeedVsIq_train.txt'))

tkinker = Tkinker(dataset)