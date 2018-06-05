# coding=utf-8
import feedparser
from numpy import *
def loadDataset():
  str1 = 'my dog has flea problems help please'
  str2 = 'maybe not take him to dog park stupid'
  str3 = 'my dalmation is so cute i love him'
  str4 = 'stop posting stupid worthless garbage'
  str5 = 'mr licks ate my streak how to stop him'
  str6 = 'quit buying worthless dog food stupid'

  arr = [str1, str2, str3, str4, str5, str6]

  postingList = [x.split(' ') for x in arr]
  classVec = [0, 1, 0, 1, 0, 1]
  return postingList, classVec

# 合并多个data数组,并去重
def createVocabList(dataSet):
  vocabSet = set([])
  for document in dataSet:
    vocabSet = vocabSet | set(document) # 求两个集合并集

  return list(vocabSet)


## inputSet 规则集合 vocabList待检测集合; 如果input中的单词在vocabList中出现了记为1
def setOfWords2Vec(vocabList, inputSet):
  returnVec = [0] * len(vocabList)

  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] = 1
    else:
      print "the word: %s is not in my Vocabulary" % word
  return returnVec

def testing0():
  listOPost, listClasses = loadDataset()
  myVocabList = createVocabList(listOPost)
  print myVocabList
  print setOfWords2Vec(myVocabList, listOPost[0])

def bagOfWords2VecMN(vocabList, inputSet):
  returnVec = [0] * len(vocabList)
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] += 1

  return returnVec


def trainNB0(trainMatrix, trainCategory):
  numTrainDocs = len(trainMatrix)  # 样本数
  numWords = len(trainMatrix[0])  # 单词总数
  pAbusive = sum(trainCategory) / float(numTrainDocs)  # ? 这里虽然是.5但是为什么是这么计算来的呢? 我觉得写成 sum(trainCategory) / len(trainCategory) 应该好一点
  p0Num = zeros(numWords)  # 标记为0的样本中, 每个单词出现的总数
  p1Num = zeros(numWords)  # 标记为1的样本中, 每个单词出现的总数
  p0Denom = 0.0
  p1Denom = 0.0

  for i in range(numTrainDocs):
    if trainCategory[i] == 1:
      p1Num += trainMatrix[i]  # 累加每个单词出现的次数
      p1Denom += sum(trainMatrix[i])  # 累加单词出现的总数

    else:
      p0Num += trainMatrix[i]  # 累加每个单词出现的次数
      p0Denom += sum(trainMatrix[i])  # 累加单词出现的总数

  print "p0Denom: %f" % p0Denom  # 24.0 标记为0的样本中, 出现的单词总数
  print "p1Denom: %f" % p1Denom  # 19.0 标记为1的样本中, 出现的单词总数

  p1Vect = p1Num / p1Denom
  p0Vect = p0Num / p0Denom

  return p0Vect, p1Vect, pAbusive

# 在trainNB0的基础上做了些优化。
def trainNB1(trainMatrix, trainCategory):
  numTrainDocs = len(trainMatrix)  # 样本数
  numWords = len(trainMatrix[0])  # 单词总数
  pAbusive = sum(trainCategory) / float(numTrainDocs)  # ? 这里虽然是.5但是为什么是这么计算来的呢? 我觉得写成 sum(trainCategory) / len(trainCategory) 应该好一点
  # 为了避免出现0的情况,这里将所有的值初始化为1.并将分母初始化为2
  p0Num = ones(numWords)  # 标记为0的样本中, 每个单词出现的总数
  p1Num = ones(numWords)  # 标记为1的样本中, 每个单词出现的总数
  p0Denom = 2.0
  p1Denom = 2.0

  for i in range(numTrainDocs):
    if trainCategory[i] == 1:
      p1Num += trainMatrix[i]  # 累加每个单词出现的次数
      p1Denom += sum(trainMatrix[i])  # 累加单词出现的总数

    else:
      p0Num += trainMatrix[i]  # 累加每个单词出现的次数
      p0Denom += sum(trainMatrix[i])  # 累加单词出现的总数

  print "p0Denom: %f" % p0Denom  # 24.0 标记为0的样本中, 出现的单词总数
  print "p1Denom: %f" % p1Denom  # 19.0 标记为1的样本中, 出现的单词总数

  # 考虑到多个小数相乘的时候,可能会造成小数溢出,直接为0,这里取了概率的ln(p)
  p1Vect = log(p1Num / p1Denom)
  p0Vect = log(p0Num / p0Denom)

  return p0Vect, p1Vect, pAbusive

def testing1():
  listOPost, listClasses = loadDataset()
  myVocabList = createVocabList(listOPost)
  trainMat = []
  for postinDoc in listOPost:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

  print "trainMat: %s" % str(trainMat)
  p0V, p1V, pAb = trainNB1(trainMat, listClasses)
  # 根据p1V得到的结果,可以看到最大的概率是0.157 对应的单词是(stupid)。说明这个词最能表征1这个特征

  print p0V
  print "\n"
  print p1V
  print "\n"
  print pAb

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
  # print vec2Classify
  # 这里vec2Classify也是一堆0和1, 但是这里的0,1是表示对应index的单词是否出现。如果出现了才会有概率计算, 所以这里是0不影响。
  p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # ln(a * b) = ln(a) + ln(b) 这里表示所有概率相乘
  p2 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

  # p(ci|w) = p(w|ci)/p(w)
  # 这里要naive nayes成立的条件是假设事件是条件独立的 即p(A1A2|B) = p(A1|B)*p(A2|B)
  # 这里 p(c0|w) = p(w0|c0)*p(w1|c0)...p(wn|c0)*p(c0) / p(w)
  #      p(c1|w) = p(w0|c1)*p(w1|c1)...p(wn|c1)*p(c1) / p(w)
  #  所以只需要比较 p(w0|c0)*p(w1|c0)...p(wn|c0)*p(c0) 和 p(w0|c1)*p(w1|c1)...p(wn|c1)*p(c1)
  print "p1: %f" % p1
  print "p2: %f" % p2
  if p1 > p2:
    return 1
  else:
    return 0

def textParse(bigString):
  import re
  listOfTokens = re.split(r'\W*', bigString)
  return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 交叉验证
def spamTest():
  docList = []
  classList = []
  fullText = []

  """
  从原始文件中,构建数据集
  """
  for i in range(1, 26):
    wordList = textParse(open('email/spam/%d.txt' % i).read())
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(1)
    wordList = textParse(open('email/ham/%d.txt' % i).read())
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(0)

  vocabList = createVocabList(docList)
  trainingSet = range(50)
  testSet = []
  """
  traingingSet中,随机取10个出来中测试集
  """
  for i in range(10):
    randIndex = int(random.uniform(0, len(trainingSet)))
    testSet.append(trainingSet[randIndex])
    del(trainingSet[randIndex])

  trainMat = []
  trainClasses = []
  for docIndex in trainingSet:
    trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
    trainClasses.append(classList[docIndex])

  p0V, p1V, pSpam = trainNB1(array(trainMat), array(trainClasses))

  errorCount = 0

  for docIndex in testSet:
    wordVector = setOfWords2Vec(vocabList, docList[docIndex])
    if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
      errorCount += 1

  print 'the error rate is: ', float(errorCount) / len(testSet)


def calcMostFreq(vocabList, fullText):
  import operator
  freqDict = {}
  for token in vocabList:
    freqDict[token] = fullText.count(token)
  sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)

  return sortedFreq[:30]

def localWords(feed1, feed0):

  docList = []
  classList = []
  fullText = []
  minLen = min(len(feed1['entries']), len(feed0['entries']))
  for i in range(minLen):
    wordList = textParse(feed1['entries'][i]['summary'])
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(1)
    wordList = textParse(feed0['entries'][i]['summary'])
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(0)

  vocabList = createVocabList(docList)
  top30Wrods = calcMostFreq(vocabList, fullText)
  trainingSet = range(2 * minLen)
  testSet = []
  for pairW in top30Wrods:
    if pairW[0] in vocabList:
      vocabList.remove(pairW[0])

  for i in range(20):
    randIndex = int(random.uniform(0, len(trainingSet)))
    testSet.append(trainingSet[randIndex])
    del(trainingSet[randIndex])

  trainMat = []
  traingClasses = []

  for docIndex in trainingSet:
    trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
    traingClasses.append(classList[docIndex])

  p0V, p1V, pSpam = trainNB1(array(trainMat), array(traingClasses))
  errorCount = 0
  for docIndex in testSet:
    wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
    if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
      errorCount += 1

  print 'the error rate is: ', float(errorCount) / len(testSet)
  return vocabList, p0V, p1V


def testingNB():
  listOPosts, listClasses = loadDataset()
  myVocabList = createVocabList(listOPosts)
  trainMat = []
  for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

  p0V, p1V, pAb = trainNB1(trainMat, listClasses)

  testEntry = ['love', 'my', 'dalmation', 'stupid']
  thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
  print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


def getTopWords(ny, sf):
  import operator
  ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
  sf = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')
  vocabList, p0V, p1V = localWords(ny, sf)
  topNy = []
  topSf = []

  for i in range(len(p0V)):
    if p0V[i] > -6.0:
      topSf.append((vocabList[i], p0V))
    if p1V[i] > -6.0:
      topNy.append((vocabList[i], p1V))


  sortedNY = sorted(topNy, key=lambda pair:pair[1], reverse=True)
  sortedSF = sorted(topSf, key=lambda pair:pair[1], reverse=True)

  for item in sortedNY:
    print item[0]

  print '------------------'

  for item in sortedSF:
    print item[0]



if __name__ == '__main__':
  #testingNB()
  #spamTest()
  ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
  sf = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')

  # 用bayes分类, 判断信息来自哪个seed
  #localWords(ny, sf)

  getTopWords(ny, sf)


