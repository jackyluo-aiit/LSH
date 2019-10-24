import math
import time

import numpy as np
import operator
import itertools
import matplotlib.pyplot as plt
import scipy.interpolate

from isort.utils import union


def accessFileToShingleMat(filename="LSH_data.txt"):
    file = open(filename)
    print("File name:", file.name)

    filelist = file.readlines()
    print("Read lines:", filelist.__len__())
    # print("Read contents:")
    index = 1

    dictlist = []
    doccount = []
    for line in filelist:
        filedict = {}
        line = line.strip()
        doc, word, occur = line.split(',')
        if doccount.__len__() != 0:
            if int(doc) not in doccount:
                doccount.append(int(doc))
        else:
            doccount.append(int(doc))
        filedict["doc"] = int(doc)
        filedict["word"] = int(word)
        filedict["occur"] = int(occur)
        dictlist.append(filedict)


    # print("The number of doc:",doccount.__len__())
    # print("doc:")
    # doccount = sorted(doccount)
    # for d in doccount:
    #     print(d)

    # dictlist.sort(key=operator.itemgetter('doc'))
    # for doc, items in itertools.groupby(dictlist,key = operator.itemgetter('doc')):
    #     if doc == 1:
    #         print(doc)
    #         for i in items:
    #             print(i)
    wordcount = max(dictlist, key=lambda x: x['word']).get('word')
    doccount = doccount.__len__()
    print("the max word:", wordcount)
    print("the min word:", min(dictlist, key=lambda x: x['word']).get('word'))
    print("the number of doc:", doccount)

    a = np.zeros((wordcount, doccount))
    for items in dictlist:
        a[items['word'] - 1][items['doc'] - 1] = 1

    print("Shingle Matrix:\n", a)
    print("Shingle Matrix.shape:", a.shape)
    return a


def accessFileToNormalMat(filename="LSH_data.txt"):
    file = open(filename)
    print("File name:", file.name)

    filelist = file.readlines()
    print("Read lines:", filelist.__len__())
    # print("Read contents:")
    index = 1

    dictlist = []
    doccount = []
    for line in filelist:
        filedict = {}
        line = line.strip()
        doc, word, occur = line.split(',')
        if doccount.__len__() != 0:
            if int(doc) not in doccount:
                doccount.append(int(doc))
        else:
            doccount.append(int(doc))
        filedict["doc"] = int(doc)
        filedict["word"] = int(word)
        filedict["occur"] = int(occur)
        dictlist.append(filedict)

    wordcount = max(dictlist, key=lambda x: x['word']).get('word')
    doccount = doccount.__len__()
    print("the max word:", wordcount)
    print("the min word:", min(dictlist, key=lambda x: x['word']).get('word'))
    print("the number of doc:", doccount)

    a = np.zeros((wordcount, doccount))
    for items in dictlist:
        a[items['word'] - 1][items['doc'] - 1] = items['occur']

    print("Shingle Matrix:\n", a)
    print("Shingle Matrix.shape:", a.shape)
    return a


def jaccardSimilarityFromTwoCol(s1, s2):
    similarcount = float(sum(s1==s2))
    similarcount = similarcount-float(sum(s1+s2==0))
    return (similarcount,float(similarcount/sum(s1+s2!=0)))


def jaccardSimilarityFromOccurance(doc, signatureMat):
    docindex = doc - 1
    similarPair = {}
    for s2 in range(np.shape(signatureMat)[1]):
        if docindex == s2:
            continue
        else:
            similarcount = float(sum(signatureMat[:, docindex] == signatureMat[:, s2]))
            similarcount = similarcount-float(sum(signatureMat[:, docindex]+signatureMat[:, s2]==0))
            similarPair[(doc, s2 + 1)] = (similarcount,
                                          float(
                                              similarcount / sum(signatureMat[:, docindex] + signatureMat[:, s2] != 0)))
    L = sorted(similarPair.items(),key = lambda item:item[1][1],reverse=True)
    L = L[:100]
    sortedSimilarPair = {}
    for l in L:
        sortedSimilarPair[l[0]] = l[1]
    return sortedSimilarPair


def minhashing(bm, permutation):
    retRow = np.zeros(np.shape(bm)[1])
    for i in range(len(permutation)):
        temp = bm[permutation.index(i + 1), :].getA()[0].nonzero()[
            0]  # select out a list of index of cell that in the row[i], and contains nonzero value.
        if len(temp) != 0:
            for index in temp:
                if retRow[index] == 0:
                    retRow[index] = i + 1  # the i start from 0, and there is no word 0.
    return retRow


def signatureMatrix(bm, minhashNum=100):
    from random import shuffle
    retMatrix = np.zeros((minhashNum, np.shape(bm)[1]))
    permutation = list(range(1, np.shape(bm)[0] + 1))
    # print(permutation)
    for i in range(minhashNum):
        shuffle(permutation)
        retMatrix[i, :] = minhashing(bm, permutation)
    return retMatrix


def LSH(signatureMat, bands, doc):
    docindex = doc - 1
    rowOfBand = math.ceil(float(np.shape(signatureMat)[0]) / bands)
    similarPair = {}
    for band in range(int(bands)):
        rowInBand = signatureMat[band * rowOfBand:min(np.shape(signatureMat)[0], (band + 1) * rowOfBand), :]
        for j in range(np.shape(rowInBand)[1]):
            if j == docindex:
                continue
            if sum(rowInBand[:, docindex] == rowInBand[:, j]) == len(rowInBand[:, docindex]):
                if (doc, j + 1) not in similarPair.keys():
                    similarPair[(doc, j + 1)] = 1
                else:
                    similarPair[(doc, j + 1)] += 1
    # for key in similarPair.keys():
    #     value = similarPair.get(key)
    #     value = (value, value / bands)
    #     similarPair[key] = value

    LSHjaccard = {}
    for key in similarPair.keys():
        doc, s2 = key
        LSHjaccard[doc,s2] = jaccardSimilarityFromTwoCol(signatureMat[:,doc-1],signatureMat[:,s2-1])

    L = sorted(LSHjaccard.items(), key=lambda item: item[1][1], reverse=True)
    L = L[:100]
    sortedSimilarPair = {}
    for l in L:
        sortedSimilarPair[l[0]] = l[1]
    return sortedSimilarPair


def checkIntersection(dict1, dict2):
    if len(dict1) != len(dict2):
        print('dict1.length: ',len(dict1))
        print('dict2.length: ', len(dict2))
        return 'They have different length'
    count = 0
    for k1 in dict1.keys():
        if k1 in dict2.keys():
            count += 1

    return count


def plotCarve(m):
    for b in range(1, m):
        r = m // b
        if r*b!=m:
            continue
        x = np.arange(0, 1, 0.01)
        y = 1 - (1 - x ** r) ** b
        plt.xlim(0, 1)
        plt.xlabel("s")
        plt.ylabel("prob")
        plt.plot(x, y, linestyle="-",label="r = %s; b = %s"%(r,b))
        plt.legend()
        plt.title("plot")
    plt.show()


if __name__ == '__main__':
    start = time.time()
    # boolMat = np.mat(boolMat)
    # print(signatureMatrix(boolMat)[0:3])
    # test:
    # boolMat = np.matrix('1 0 1 0;1 0 0 1;0 1 0 1; 0 1 0 1; 0 1 0 1;1 0 1 0;1 0 1 0')
    # permu = [[2, 3, 6, 5, 7, 1, 4], [3, 1, 7, 2, 5, 6, 4], [7, 2, 6, 5, 1, 4, 3]]
    # for p in permu:
    #     print(minhashing(boolMat,p))

    # boolMat = accessFileToShingleMat("LSH_data.txt")
    # boolMat = np.mat(boolMat)
    # signatureMat = signatureMatrix(boolMat, 500)
    # # print(signatureMatrix(boolMat))
    # normalMat = accessFileToNormalMat("LSH_data.txt")
    # print("Using LSH:", LSH(signatureMat, 40, 2))
    # print("Without using LSH:", jaccardSimilarityFromOccurance(2, normalMat))
    # print("intersection: ", (checkIntersection(LSH(signatureMat, 40, 2), jaccardSimilarityFromOccurance(2, normalMat))))
    # # plotCarve(20, 20)
    # print('Time taken: {} secs\n'.format(time.time() - start))


    plotCarve(500)
    # print(boolMat[0,:].getA()[0].nonzero())
    # print(jaccardSimilarityFromTwoCol(boolMat[:, 1], boolMat[:, 0]))

# print(jaccardSimilarityFromTwoCol(np.array([1, 0, 0, 1, 0]), np.array([0, 0, 1, 1, 0])))

# docnums = itertools.groupby(filelist,key=lambda x:x[2])
# for k,g in docnums:
#     print(g)
