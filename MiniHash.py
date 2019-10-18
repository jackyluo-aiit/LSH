import numpy as np
import operator
import itertools


def accessFile(filename="LSH_data.txt"):
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
        # if index%1000==0:
        #     print(filedict)
        # index+=1

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


def jaccardSimilarityFromTwoCol(s1, s2):
    return float(sum(s1 + s2 == 2) / sum(s1 + s2 != 0))


def minhashing(bm, permutation):
    retRow = np.zeros(np.shape(bm)[1])
    for i in range(len(permutation)):
        temp = bm[permutation.index(i+1),:].getA()[0].nonzero()[0]
        if len(temp)!=0:
            for index in temp:
                if retRow[index] == 0:
                    retRow[index]=i+1
    return retRow



def signatureMatrix(bm, minhashNum=100):
    from random import shuffle
    retMatrix = np.zeros((minhashNum,np.shape(bm)[1]))
    permutation = list(range(1,np.shape(bm)[0]+1))
    # print(permutation)
    for i in range(minhashNum):
        shuffle(permutation)
        retMatrix[i,:] = minhashing(bm,permutation)
    return retMatrix


if __name__ == '__main__':
    boolMat = accessFile()
    # boolMat = np.matrix('1 0 1 0;1 0 0 1;0 1 0 1; 0 1 0 1; 0 1 0 1;1 0 1 0;1 0 1 0')
    # permu = [[2, 3, 6, 5, 7, 1, 4], [3, 1, 7, 2, 5, 6, 4], [7, 2, 6, 5, 1, 4, 3]]
    # for p in permu:
    #     print(minhashing(boolMat,p))
    boolMat = np.mat(boolMat)
    print(signatureMatrix(boolMat)[0:3])

    # print(boolMat[0,:].getA()[0].nonzero())
    # print(jaccardSimilarityFromTwoCol(boolMat[:, 1], boolMat[:, 0]))

# print(jaccardSimilarityFromTwoCol(np.array([1, 0, 0, 1, 0]), np.array([0, 0, 1, 1, 0])))

# docnums = itertools.groupby(filelist,key=lambda x:x[2])
# for k,g in docnums:
#     print(g)
