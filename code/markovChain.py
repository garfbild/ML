import numpy as np
import random

class markovChain:
    def __init__(self,n = 1,alpha = 0):
        self._matrix = np.zeros([n,n])
        self._alpha = alpha
        self._n = n

    def setMatrix(self,m):
        self._matrix = m
        self._n = len(m[0])

    def normalise(self):
        for i in range(self._n):
            s = 0
            for j in range(self._n):
                s += self._matrix[i,j]
            if s != 0:
                for j in range(self._n):
                    self._matrix[i,j] = self._matrix[i,j]/s

    def nextNode(self,i):
        a = random.random()
        if a < self._alpha:
            return random.randint(0, self._n)
        s = 0
        r = random.random()
        for j in range(self._n):
            s += self._matrix[i,j]
            if r < s:
                return j
        return -1

    def generateText(self,n):
        S = []
        for i in range(n):
            s = []
            start = random.randint(0, self._n)
            s.append(start)
            n = self.nextNode(start)
            while n != -1:
                s.append(n)
                n = self.nextNode(n)
            S.append(s)
        return S


import csv

tweets = []
with open('code\Data\@tweetmynuts4_user_tweets.csv', newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        try:
            if row[1][0:2] != "RT":
                tweets.append(row[1])
        except:
            x = 1
tweets = tweets[1:]


import string

words = []
dictionary = {}
for tweet in tweets:
    for word in tweet.split():
        if word[0] != "@" and word[0:5] != "https" and word not in words:
            words.append(word.translate(str.maketrans('', '', string.punctuation)))
            dictionary[word.translate(str.maketrans('', '', string.punctuation))] = len(words)-1

print(words)
n = len(words)
print(dictionary)
adjacencyMatrix = np.zeros([n,n])
for tweet in tweets:
    spleet = tweet.split()
    for i in range(len(spleet)-1):
        word = spleet[i]
        if word[0] != "@" and word[0:5] != "https":
            try:
                x = dictionary[word]
                y = dictionary[spleet[i+1]]
                adjacencyMatrix[x,y] += 1
            except:
                z = 1


M = markovChain(n,0.01)
M.setMatrix(adjacencyMatrix)
M.normalise()
newTweets = M.generateText(50)
for tweet in newTweets:
    print(' '.join([words[i] for i in tweet]))
