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
    best_mlp_missed = []
    ae_1_mlp_missed = []
    ae_2_mlp_missed = []
    ae_3_mlp_missed = []
    for testing in range(10):
        mlp_0_missed = []
        mlp_1_missed = []
        mlp_2_missed = []
        print("Fold: ",testing)
        training_set = []

        testing_set = chunked_data[testing]
        # make example set
        for train in range(10):
            if train != testing:
                for x in range(len(chunked_data[train])):
                    training_set.append(chunked_data[train][x])

        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        if use_regression:
            # train multi layer perceptrons
            mlp_0 = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp_0.tune(training_set[:validation_index], training_set[validation_index:], 0, [], 10, 100)
            mlp_1 = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp_1.tune(training_set[:validation_index], training_set[validation_index:], 1, [], 10, 100)
            mlp_2 = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp_2.tune(training_set[:validation_index], training_set[validation_index:], 2, [], 10, 100)
            # test multi layer perceptrons
            mlp_0_missed = ms.testRegressor(mlp_0, testing_set)
            mlp_1_missed = ms.testRegressor(mlp_1, testing_set)
            mlp_2_missed = ms.testRegressor(mlp_2, testing_set)

            lowest_error_mlp = mlp_0
            lowest_error = mlp_0_missed
            print("Average absolute error for 0-layer MLP: ", ms.getMean(mlp_0_missed,len(mlp_0_missed)))
            print("Average absolute error for 1-layer MLP: ", ms.getMean(mlp_1_missed, len(mlp_1_missed)))
            print("Average absolute error for 2-layer MLP: ", ms.getMean(mlp_2_missed, len(mlp_2_missed)))
            if ms.getMean(mlp_1_missed,len(mlp_1_missed)) < ms.getMean(mlp_0_missed,len(mlp_0_missed)):
                lowest_error_mlp = mlp_1
                lowest_error = mlp_1_missed
            if ms.getMean(mlp_2_missed,len(mlp_2_missed)) < ms.getMean(lowest_error,len(lowest_error)):
                lowest_error_mlp = mlp_2
                lowest_error = mlp_2_missed
            best_mlp_missed.append(lowest_error)
            print("Selected MLP with",len(lowest_error_mlp.hidden_layers), "layers")
            # train autoencoders
            ae_1 = ffn.FeedforwardNetwork(len(chunked_data[0][0])-1, clss_list, "autoencoder", True, False)
            ae_1.regularizeAutoencoder(20)
            ae_1.tune(training_set[:validation_index], training_set[validation_index:], 1, [], 10, 150)
            # Autoencoder 1 stacked MLP
            ae_1_mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ae_1.addFFNetwork(ae_1_mlp, True, len(lowest_error_mlp.hidden_layers), training_set[:validation_index], training_set[validation_index:], 10, 100)

            ae_2 = ffn.FeedforwardNetwork(len(chunked_data[0][0]) - 1, clss_list, "autoencoder", True, False)
            ae_2.regularizeAutoencoder(20)
            ae_2.tune(training_set[:validation_index], training_set[validation_index:], 2, [], 10, 150)
            # Autoencoder 2 stacked MLP
            ae_2_mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ae_2.addFFNetwork(ae_2_mlp, True, len(lowest_error_mlp.hidden_layers), training_set[:validation_index],training_set[validation_index:], 10, 100)

            ae_3 = ffn.FeedforwardNetwork(len(chunked_data[0][0]) - 1, clss_list, "autoencoder", True, False)
            ae_3.regularizeAutoencoder(20)
            ae_3.tune(training_set[:validation_index], training_set[validation_index:], 3, [], 10, 150)
            # Autoencoder 3 stacked MLP
            ae_3_mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ae_3.addFFNetwork(ae_3_mlp, True, len(lowest_error_mlp.hidden_layers), training_set[:validation_index],training_set[validation_index:], 10, 100)

            # test autoencoders
            ae_1_mlp_missed.append(ms.testRegressor(ae_1, testing_set))
            ae_2_mlp_missed.append(ms.testRegressor(ae_2, testing_set))
            ae_3_mlp_missed.append(ms.testRegressor(ae_3, testing_set))
            #print(ms.getMean(best_mlp_missed[0], len(best_mlp_missed[0])), ms.getMean(ae_3_mlp_missed[0], len(ae_3_mlp_missed[0])))

        else:
            # train multi layer perceptrons
            mlp_0 = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            mlp_0.tune(training_set[:validation_index], training_set[validation_index:], 0, [], 10, 100)
            mlp_1 = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            mlp_1.tune(training_set[:validation_index], training_set[validation_index:], 1, [], 10, 100)
            #mlp_2 = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            #mlp_2.tune(training_set[:validation_index], training_set[validation_index:], 2, [], 10, 100)
            # test multi layer perceptrons
            mlp_0_missed = ms.testProbabilisticClassifier(mlp_0, testing_set)
            #mlp_1_missed = ms.testProbabilisticClassifier(mlp_1, testing_set)
            #mlp_2_missed = ms.testProbabilisticClassifier(mlp_2, testing_set)

            lowest_error_mlp = mlp_0
            lowest_error = mlp_0_missed
            print("0-1 loss for 0-layer MLP: ", ms.getMeanValofTuple(mlp_0_missed,0))
            print("0-1 loss for 1-layer MLP: ", ms.getMeanValofTuple(mlp_1_missed,0))
            print("0-1 loss for 2-layer MLP: ", ms.getMeanValofTuple(mlp_2_missed,0))
            if ms.getMeanValofTuple(mlp_1_missed,0) < ms.getMeanValofTuple(mlp_0_missed,0):
                lowest_error_mlp = mlp_1
                lowest_error = mlp_1_missed
            if ms.getMeanValofTuple(mlp_2_missed,0) < ms.getMeanValofTuple(lowest_error,0):
                lowest_error_mlp = mlp_2
                lowest_error = mlp_2_missed
            best_mlp_missed.append(lowest_error)
            print("Selected MLP with", len(lowest_error_mlp.hidden_layers), "layers")
            # train autoencoders
            ae_1 = ffn.FeedforwardNetwork(len(chunked_data[0][0])-1, clss_list, "autoencoder", True, False)
            ae_1.regularizeAutoencoder(20)
            ae_1.tune(training_set[:validation_index], training_set[validation_index:], 1, [], 10, 150)
            # Autoencoder 1 stacked MLP
            ae_1_mlp = ffn.FeedforwardNetwork(1, clss_list, "classification", True, True)
            ae_1.addFFNetwork(ae_1_mlp, True, len(lowest_error_mlp.hidden_layers), training_set[:validation_index], training_set[validation_index:], 10, 100)

            # test algorithms
            mlp_0_missed.append(ms.testProbabilisticClassifier(mlp_0, testing_set))
            mlp_1_missed.append(ms.testProbabilisticClassifier(mlp_1, testing_set))
            mlp_2_missed.append(ms.testProbabilisticClassifier(mlp_2, testing_set))
    if use_regression:
        ms.compareRegressors(best_mlp_missed, ae_1_mlp_missed, "MLP", "MLP stacked on 1-layer Autoencoder")
        ms.compareRegressors(best_mlp_missed, ae_2_mlp_missed, "MLP", "MLP stacked on 2-layer Autoencoder")
        ms.compareRegressors(best_mlp_missed, ae_3_mlp_missed, "MLP", "MLP stacked on 2-layer Autoencoder")

        ms.compareRegressors(ae_1_mlp_missed, ae_2_mlp_missed, "MLP stacked on 1-layer Autoencoder", "MLP stacked on 2-layer Autoencoder")
        ms.compareRegressors(ae_1_mlp_missed, ae_3_mlp_missed, "MLP stacked on 1-layer Autoencoder", "MLP stacked on 3-layer Autoencoder")

        ms.compareRegressors(ae_2_mlp_missed, ae_3_mlp_missed, "MLP stacked on 2-layer Autoencoder","MLP stacked on 3-layer Autoencoder")
    else:
        ms.compareProbabilisticClassifiers(cn_rbfn_missed, c_rbfn_missed, "CNN RBF", "k-means RBF")
        print("CNN RBF", "k-means RBF", "convergence time")
        ms.pairedTTest(cn_rbfn_time, c_rbfn_time, 0.05)
        ms.compareProbabilisticClassifiers(cn_rbfn_missed, m_rbfn_missed, "CNN RBF", "PAM RBF")
        print("CNN RBF", "PAM RBF", "convergence time")
        ms.pairedTTest(cn_rbfn_time, m_rbfn_time, 0.05)
        ms.compareProbabilisticClassifiers(cn_rbfn_missed, mlp_0_missed, "CNN RBF", "0-layer MLP")
        print("CNN RBF", "0-layer MLP", "convergence time")
        ms.pairedTTest(cn_rbfn_time, mlp_0_time, 0.05)
        ms.compareProbabilisticClassifiers(cn_rbfn_missed, mlp_1_missed, "CNN RBF", "1-layer MLP")
        print("CNN RBF", "1-layer MLP", "convergence time")
        ms.pairedTTest(cn_rbfn_time, mlp_1_time, 0.05)
        ms.compareProbabilisticClassifiers(cn_rbfn_missed, mlp_2_missed, "CNN RBF", "2-layer MLP")
        print("CNN RBF", "2-layer MLP", "convergence time")
        ms.pairedTTest(cn_rbfn_time, mlp_2_time, 0.05)

        ms.compareProbabilisticClassifiers(c_rbfn_missed, m_rbfn_missed, "k-means RBF", "PAM RBF")
        print("k-means RBF", "PAM RBF", "convergence time")
        ms.pairedTTest(c_rbfn_time, m_rbfn_time, 0.05)
        ms.compareProbabilisticClassifiers(c_rbfn_missed, mlp_0_missed, "k-means RBF", "0-layer MLP")
        print("k-means RBF", "0-layer MLP", "convergence time")
        ms.pairedTTest(c_rbfn_time, mlp_0_time, 0.05)
        ms.compareProbabilisticClassifiers(c_rbfn_missed, mlp_1_missed, "k-means RBF", "1-layer MLP")
        print("k-means RBF", "1-layer MLP", "convergence time")
        ms.pairedTTest(c_rbfn_time, mlp_1_time, 0.05)
        ms.compareProbabilisticClassifiers(c_rbfn_missed, mlp_2_missed, "k-means RBF", "2-layer MLP")
        print("k-means RBF", "2-layer MLP", "convergence time")
        ms.pairedTTest(c_rbfn_time, mlp_2_time, 0.05)

        ms.compareProbabilisticClassifiers(m_rbfn_missed, mlp_0_missed, "PAM RBF", "0-layer MLP")
        print("PAM RBF", "0-layer MLP", "convergence time")
        ms.pairedTTest(m_rbfn_time, mlp_0_time, 0.05)
        ms.compareProbabilisticClassifiers(m_rbfn_missed, mlp_1_missed, "PAM RBF", "1-layer MLP")
        print("PAM RBF", "1-layer MLP", "convergence time")
        ms.pairedTTest(m_rbfn_time, mlp_1_time, 0.05)
        ms.compareProbabilisticClassifiers(m_rbfn_missed, mlp_2_missed, "PAM RBF", "2-layer MLP")
        print("PAM RBF", "2-layer MLP", "convergence time")
        ms.pairedTTest(m_rbfn_time, mlp_2_time, 0.05)

        ms.compareProbabilisticClassifiers(mlp_0_missed, mlp_1_missed, "0-layer MLP", "1-layer MLP")
        print("0-layer MLP", "1-layer MLP", "convergence time")
        ms.pairedTTest(mlp_0_time, mlp_1_time, 0.05)
        ms.compareProbabilisticClassifiers(mlp_0_missed, mlp_2_missed, "0-layer MLP", "2-layer MLP")
        print("0-layer MLP", "2-layer MLP", "convergence time")
        ms.pairedTTest(mlp_0_time, mlp_2_time, 0.05)

        ms.compareProbabilisticClassifiers(mlp_1_missed, mlp_2_missed, "1-layer MLP", "2-layer MLP")
        print("1-layer MLP", "2-layer MLP", "convergence time")
        ms.pairedTTest(mlp_1_time, mlp_2_time, 0.05)

        print("average CNN RBF time:", ms.getMean(cn_rbfn_time, len(cn_rbfn_time)))
        print("average k-means RBF time:", ms.getMean(c_rbfn_time, len(c_rbfn_time)))
        print("average PAM RBF time:", ms.getMean(m_rbfn_time, len(m_rbfn_time)))
        print("average 0-layer MLP time:", ms.getMean(mlp_0_time, len(mlp_0_time)))
        print("average 1-layer MLP time:", ms.getMean(mlp_1_time, len(mlp_1_time)))
        print("average 2-layer MLP time:", ms.getMean(mlp_2_time, len(mlp_2_time)))

if(len(sys.argv) > 2):
    chunks, class_list = openFiles(sys.argv[1])
    uses_regression = False
    if sys.argv[2] == 'r':
        print("Using regression")
        uses_regression = True
    else:
        print("Using classification")

    print("Using k=3")
    trainAndTest(chunks, class_list, 3, uses_regression)
else:
    print("Usage:\t<dataFile.data> <r> (for regression, use any other character for classification)")