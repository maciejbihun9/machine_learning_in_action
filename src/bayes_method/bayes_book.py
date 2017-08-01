"""
We could instead ask the classifier
to give us a best guess about the class
and assign a probability estimate to that best guess.

Naïve Bayes
Pros: Works with a small amount of data, handles multiple classes
Cons: Sensitive to how the input data is prepared
Works with: Nominal values

rule for classifier:
* If p1(x, y) > p2(x, y), then the class is 1.
* If p2(x, y) > p1(x, y), then the class is 2.
"""

from numpy import *

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# return list with number of word occurances in vocabList.
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# create set of words from dataSet(documents list)
# words are anot repeated
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# parse words to the number vector
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

# computes probability of words being abusive or not.
# also the probability of the 1 or 0 class.
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
    # Numerator
    # counted occurances of words in not abusive documents
    p0Num = ones(numWords)

    # counted occurances of words in abusive documents
    p1Num = ones(numWords)
    # Denominator
    # number of words in not abusive documents
    p0Denom = 2.0
    # number of words in abusive documents
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # if abusive category document
        if trainCategory[i] == 1:
            # count abbusive words in abusive document
            p1Num += trainMatrix[i]
            # sum all words in abusive documents
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # holds propability of selecting word from vocabulary in abusive docs
    p1Vect = log(p1Num / p1Denom)  # change to log() -> P(1)

    # holds propability of selecting word from vocabulary in not abusive docs
    p0Vect = log(p0Num / p0Denom)  # change to log() -> P(0)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # lower value of the p means have worse probability
    if p1 > p0:
        return 1
    else:
        return 0