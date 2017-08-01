from numpy import *
from src.bayes_method.bayes_book import *

def spamTest():
    docList= []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('../../resources/email/spam/%d.txt' % i).read())
        # list with document words list
        docList.append(wordList)
        # list with all words from all documents
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('../../resources/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        # set with all words without repetition
        vocabList = createVocabList(docList)
    trainingSet = [i for i in range(50)]
    testSet = []
    # pick items to test
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    trainMat = []
    trainClasses = []
    # parse words from docs to numbers and add it to the trainMat
    # add also classes to the trainClasses
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        # create wordVector of numbers from test words set
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # if class computed by classifyNB method is not equal with classList then increment errorCount
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ', float(errorCount) / len(testSet))

spamTest()