from math import log
import operator
import pickle

"""
We assume that class items are always in the last column.
"""

# CREATE TREE
def chooseBestFeatureToSplit(dataSet):
    """
    Splits the dataset in place where the new datasets have the best entropy value.
    Returns feature index where value of entropy is the highest.
    """
    # get number of features without class attribute
    numFeatures = len(dataSet[0]) - 1
    # computes start dataset entropy
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # for each dataset feature
    for i in range(numFeatures):
        # get i -th column items
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # compute entropy for splitted items
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # get probability of selecting a dataset item which starts with unique value of i-th column.
            prob = len(subDataSet)/float(len(dataSet))
            # compute entropy for that new data set
            newEntropy += prob * calcShannonEnt(subDataSet)
        # compute how much info we lose after split on i-th column
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def splitDataSet(dataSet, axis, value):
    """
    :param dataSet: the dataset we’ll split,
    :param axis: the feature we’ll split on,
    :param value: value of the feature to return.
    """
    # new dataset without items that starts in axis with value
    retDataSet = []
    # for each item in dataset
    for featVec in dataSet:
        # if in item for feature under index (axis) value == value
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # cut out the feature that we split on
            # once you’ve split on a feature, you’re finished with that feature.
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcShannonEnt(dataSet):
    """
    Calculates Shannon entropy value.
    Could be described as task set disorder.
    :param dataSet: list of lists
    :return: float entropy value
    """
    numEntries = len(dataSet)
    labelCounts = {}
    # count classes occurrences
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # computes entropy value based on each item occurrences probability.
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def majorityCnt(classList):
    """
    return the class that occurs with the greatest frequency.
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    # get last column items
    classList = [example[-1] for example in dataSet]
    # if in dataset exist only one class
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # if we have only one item in dataset
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # choose best featature to split the data
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # best feature to split
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # remove that feature
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # create tree branch for each best feature value
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                return secondDict[key]
    return classLabel


# STORING OBJECTS
def storeTree(inputTree,filename):
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)