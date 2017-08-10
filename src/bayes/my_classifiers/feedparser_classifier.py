import feedparser

from src.bayes.book_classifiers.text_classifier import *


def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def getTopWords(ny,sf):
    vocabList,p0V,p1V=localWords(ny,sf)
    # get words with the highest prop
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **")
    for item in sortedNY:
        print (item[0])

def localWords(feed1,feed0):
    docList=[]; classList = []; fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        # get webpage urls summuries text
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # get webpage urls summuries text
        wordList = textParse(feed0['entries'][i]['summary'])
        # list with webpage summuries
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # vocab list with all words used in all webpages.
    vocabList = createVocabList(docList)
    # create a list of tuples that describes occurances of words.
    top30Words = calcMostFreq(vocabList,fullText)
    # remove to 30 words from vocablist
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = [i for i in range(2*minLen)]
    testSet=[]
    # create test set and remove selected items from training set.
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses = []
    # convert words to numbers and get rss sites classes
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    # for each document measure the error rate
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList,pSF,pNY = localWords(ny,sf)
top_words = getTopWords(ny, sf)
print("Vocablist: {}, pSF: {}, pNY: {}".format(vocabList,pSF,pNY))

