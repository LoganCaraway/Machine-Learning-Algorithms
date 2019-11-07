import MathAndStats as ms
import csv
import sys
import random
import math
import NearestNeighbor as nn
import Kmeans as km
import PAM as pam
import RBFNetwork as rbf
import FeedforwardNetwork as ffn

#--------------------DATA-MANIPULATION--------------------#
def openFiles(dataFile):
    lines = open(dataFile, "r").readlines()
    csvLines = csv.reader(lines)
    data = list()
    #save = None

    for line in csvLines:
        tmp = []
        for c in range(0, len(line) - 1):
            tmp.append(float(line[c]))
        if sys.argv[2] == 'r':
            tmp.append(float(line[-1]))
        else:
            tmp.append(line[-1])
        data.append(tmp)

    # remove line number from each example (first column)
    for example in range(len(data)):
        del data[example][0]

    data = ms.normalize(data)
    #print(data)
    if sys.argv[1] == "output_machine.data":
        for obs in range(len(data)):
            if int(data[obs][-1]) < 21:
                data[obs][-1] = 1
            elif int(data[obs][-1]) < 101:
                data[obs][-1] = 2
            elif int(data[obs][-1]) < 201:
                data[obs][-1] = 3
            elif int(data[obs][-1]) < 301:
                data[obs][-1] = 4
            elif int(data[obs][-1]) < 401:
                data[obs][-1] = 5
            elif int(data[obs][-1]) < 501:
                data[obs][-1] = 6
            elif int(data[obs][-1]) < 601:
                data[obs][-1] = 7
            else:
                data[obs][-1] = 8

    if (len(sys.argv) > 3) and (sys.argv[3] == "log"):
        logOutputs(data)
    # divide data into 10 chunks for use in 10-fold cross validation paired t test
    chnks = getNChunks(data, 10)
    class_list = getClasses(data)

    # get a boolean vector telling whether to use euclidean distance or hamming distance on a feature-by-feature basis
    #data_metric = getDataMetrics()

    return chnks, class_list

# divide the example set into n random chunks of approximately equal size
def getNChunks(data, n):
    # randomly shuffle the order of examples in the data set
    random.shuffle(data)
    dataLen = len(data)
    chunkLen = int(dataLen / n)
    # chunks is a list of the individual chunks
    chunks = []
    # rows are observation
    # columns are labels

    # skip along the data file chunking every chunkLen
    for index in range(0, dataLen, chunkLen):
        if (index + chunkLen) <= dataLen:
            # copy from current skip to the next
            chunk = data[index:index + chunkLen]
            # chunks is a list of the individual chunks
            chunks.append(chunk)
    # append the extra examples to the last chunk
    for i in range(n*chunkLen, dataLen):
        chunks[-1].append(data[i])
    for i in range(len(chunks)):
        print("Length of chunk: ", len(chunks[i]))
    return chunks
#--------------------DATA-MANIPULATION-END--------------------#

def logOutputs(data):
    if not sys.argv[2] == 'r':
        print("Only log outputs for regression")
        return
    for example in range(len(data)):
        temp = data[example][-1]
        del data[example][-1]
        if temp == 0:
            temp = 0.001
        data[example].append(math.log(temp))

def getClasses(data):
    if sys.argv[2] == 'r':
        return []
    classes = []
    for x in range(len(data)):
        if not data[x][-1] in classes:
            classes.append(data[x][-1])
    return classes

def trainAndTest(chunked_data, clss_list, k, use_regression):
    base_missed = []
    cnn_missed = []
    kmeans_missed = []
    pam_missed = []
    cn_rbf_missed = []
    c_rbf_missed = []
    m_rbf_missed = []
    mlp_0_missed = []
    mlp_1_missed = []
    mlp_2_missed = []
    for testing in range(10):
        training_set = []
        #testing_set = []

        testing_set = chunked_data[testing]
        # make example set
        for train in range(10):
            if train != testing:
                for x in range(len(chunked_data[train])):
                    training_set.append(chunked_data[train][x])

        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        if use_regression:
            # train algorithms
            #kNN = nn.NearestNeighbor(training_set, k, use_regression)
            #kmeans = km.KMeans(training_set[:validation_index], int(len(training_set[:validation_index])/4), use_regression, 2)
            #pm = pam.PAM(training_set[:validation_index], int(len(training_set[:validation_index])/4), use_regression, 2)
            #c_rbfn = rbf.RBFNetwork(kmeans.centroids, kmeans.clust, clss_list, use_regression, False)
            #c_rbfn.tune(training_set[:validation_index], training_set[validation_index:])
            #m_rbfn = rbf.RBFNetwork(pm.medoids, pm.clust, clss_list, use_regression, False)
            #m_rbfn.tune(training_set[:validation_index], training_set[validation_index:])
            mlp_0 = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp_0.tune(training_set[:validation_index], training_set[validation_index:], 2)
            # test algorithms

            #base_missed.append(ms.testRegressor(kNN, testing_set))
            #kmeans_missed.append(ms.testRegressor(kmeans, testing_set))
            #c_rbf_missed.append(ms.testRegressor(c_rbfn, testing_set))
            #m_rbf_missed.append(ms.testRegressor(m_rbfn, testing_set))
            #mlp_missed.append(ms.testRegressor(mlp, testing_set))
            #pam_missed.append(ms.testRegressor(pm, testing_set))
        else:
            # train algorithms
            kNN = nn.NearestNeighbor(training_set, k, use_regression)
            # RBF based on Condensed NN
            cNN = nn.NearestNeighbor(training_set, k, use_regression)
            cNN.convertToCondensed()
            cNN_clust = []
            for obs in range(len(cNN.training_set)):
                cNN_clust.append(kNN.getNeighbors(cNN.training_set[obs])[1:])
            cn_rbfn = rbf.RBFNetwork(cNN.training_set, cNN_clust, clss_list, use_regression, True)
            cn_rbfn.tune(training_set[:validation_index], training_set[validation_index:])
            # RBF based on k-means
            kmeans = km.KMeans(training_set, len(cNN.training_set), uses_regression, 2)
            c_rbfn = rbf.RBFNetwork(kmeans.centroids, kmeans.clust, clss_list, use_regression, True)
            c_rbfn.tune(training_set[:validation_index], training_set[validation_index:])
            # RBF based on PAM
            pm = pam.PAM(training_set,len(cNN.training_set), use_regression, 2)
            m_rbfn = rbf.RBFNetwork(pm.medoids, pm.clust, clss_list, use_regression, True)
            m_rbfn.tune(training_set[:validation_index], training_set[validation_index:])
            # 0 layer MLP
            mlp_0 = ffn.FeedforwardNetwork(1, clss_list, "classification", True, True)
            mlp_0.tune(training_set[:validation_index], training_set[validation_index:], 0)
            # 1 layer MLP
            mlp_1 = ffn.FeedforwardNetwork(1, clss_list, "classification", True, True)
            mlp_1.tune(training_set[:validation_index], training_set[validation_index:], 1)
            # 2 layer MLP
            mlp_2 = ffn.FeedforwardNetwork(1, clss_list, "classification", True, True)
            mlp_2.tune(training_set[:validation_index], training_set[validation_index:], 2)
            # test algorithms
            cn_rbf_missed.append(ms.testClassifier(cn_rbfn, testing_set))
            c_rbf_missed.append(ms.testClassifier(c_rbfn, testing_set))
            m_rbf_missed.append(ms.testClassifier(m_rbfn, testing_set))
            mlp_0_missed.append(ms.testClassifier(mlp_0_missed, testing_set))
            mlp_1_missed.append(ms.testClassifier(mlp_1_missed, testing_set))
            mlp_2_missed.append(ms.testClassifier(mlp_2_missed, testing_set))
    if use_regression:
        ms.compareRegressors(c_rbf_missed, m_rbf_missed, "k-means RBF", "PAM RBF")
        ms.compareRegressors(kmeans_missed, pam_missed, "k-means", "PAM")
        #ms.compareRegressors(base_missed, kmeans_missed, "K-Means Clustering")
        #ms.compareRegressors(base_missed, rbf_missed, "RBF")
        #ms.compareRegressors(base_missed, mlp_missed, "MLP")
        #ms.compareRegressors(base_missed, pam_missed, "PAM")
    else:
        pass
        #ms.compareClassifiers(base_missed, cnn_missed, "Condensed Nearest Neighbor")
        #ms.compareClassifiers(base_missed, enn_missed, "Edited Nearest Neighbor")
        #ms.compareClassifiers(base_missed, kmeans_missed, "K-Means Clustering")
        #ms.compareClassifiers(base_missed, rbf_missed, "RBF")
        #ms.compareClassifiers(base_missed, mlp_missed, "MLP")
        #compareClassifiers(base_missed, pam_missed, "PAM")

if(len(sys.argv) > 2):
    chunks, class_list = openFiles(sys.argv[1])
    uses_regression = False
    if sys.argv[2] == 'r':
        print("Using regression")
        uses_regression = True
    else:
        print("Using classification")

    #class_list = getClasses(chunks)
    print("Using k=3")
    trainAndTest(chunks, class_list, 3, uses_regression)
    k_tenth = int( float(len(chunks[0][0])-1) / 10 )
    if (k_tenth == 0) or (k_tenth == 1):
        k_tenth = 2
    print("Using k is one tenth of the number of features (rounded up to 2). k =",k_tenth)
    #trainAndTest(chunks, k_tenth, uses_regression)
    k_root = int( pow(float(len(chunks[0][0])-1), 0.5) )
    if (k_root == 0) or (k_root == 1):
        k_root = 2
    print("Using k is the square root of the number of features (rounded up to 2). k =", k_root)
    #trainAndTest(chunks, k_root, uses_regression)
else:
    print("Usage:\t<dataFile.data> <r> (for regression, use any other character for classification)")