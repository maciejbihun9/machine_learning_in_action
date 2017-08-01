
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
from src.bayes_method.bayes_book import *

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



def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['stupid', 'problems', 'help', 'please', 'is', 'so', 'cute']
    # get testEntry positions in vocabulary
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)

# matrix with post words position in vocabulary
# knowing what word is at the position in the list.
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

p0V, p1V, pAb = trainNB0(trainMat, listClasses)
print("p0V; {}".format(p0V))
print("p1V; {}".format(p1V))
print("pAb; {}".format(pAb))

testingNB()

print(myVocabList)