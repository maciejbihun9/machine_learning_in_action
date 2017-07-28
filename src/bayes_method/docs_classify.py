
"""
* We can treat the absence of a word or a statement as a feature.

* We can assume that each word is independent(naive assumption),
* all words are equally important
- We know that it is not true, because certainly words are more likely to see after another that others,
- We also know that we do not have to read all words
- besides that naive bayes works well in practice.
*

"""
from numpy import *

# return word set and classes attached to each list
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
        'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', \
        'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', \
        'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
        'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList,classVec

# return list of words using set sum that eliminates duplicated words.
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# if word from input set exists in vocablist then desc it as a 1 symbol.
# establish at which position word exists in the vocabulary
def setOfWords2Vec(vocabList, inputSet):
    """
    :param vocabList: vocabulary list
    :param inputSet: words vector
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    else:
        print("the word: {} is not in my Vocabulary!".format(word))
    return returnVec

def trainNB0(trainMatrix: matrix ,trainCategory):
    """

    :param trainMatrix: matrix with documents,
    :param trainCategory: vector with document labels.
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # propability of selecting abusive document
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        # if abusive category document
        if trainCategory[i] == 1:
            # count occurances of word in abbusive document
            p1Num += trainMatrix[i]

            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # change to log()
    p0Vect = p0Num / p0Denom  # change to log()
    return p0Vect, p1Vect, pAbusive


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)

# matrix with post words position in vocabulary
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

p0V, p1V, pAb = trainNB0(trainMat, listClasses)
print("Fuck")
print(myVocabList)