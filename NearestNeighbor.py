#Created 9/20/2019
#Logan Caraway
#Created for CSCI447 Machine Learning

import MathAndStats as ms
import random
import copy

class NearestNeighbor:

    # uses_euclidean is a boolean array for each attribute. If not euclidean distance metric: manhattan distance metric
    # uses_regression is a boolean value. If not regression: classification
    def __init__(self, data, k, uses_regression):
        self.k = k
        self.uses_regression = uses_regression
        self.training_set = data

    # start training set as only holding a single element from the original training set
    # loop through the original training set:
    #     if an item is misclassified, add it to the new training set and restart at the beginning of the original training set
    # continue until the new training set does not change or the probability of misclassification stops decreasing
    def convertToCondensed(self):
        if self.uses_regression:
            print("Condensed Nearest Neighbor is only set up for classification, not regression")
            return

        print("Converting k-Nearest Neighbor to Condensed Nearest Neighbor")
        # the training set is going to be modified, so we need a temporary place to hold the training_set
        original_data = copy.deepcopy(self.training_set)
        random.shuffle(original_data)
        # start new_data with a random item from the original data set
        self.training_set = []
        for i in range(self.k):
            self.training_set.append(copy.deepcopy(original_data[i]))

        obs_num = 0
        #prev_missrate = 1.0
        while True:
            correct_class = original_data[obs_num][len(original_data[0])-1]
            predicted_class = self.classify(original_data[obs_num])
            # if misclassified, add to the new training set (if the training set doesn't already contain it)
            if (correct_class != predicted_class) and not (original_data[obs_num] in self.training_set):
                self.training_set.append(copy.deepcopy(original_data[obs_num]))
                obs_num = -1
            obs_num += 1
            if obs_num == len(original_data):
                break
        print("Original size:", len(original_data))
        print("Reduced size:", len(self.training_set))

    # similar to condensed nearest neighbor, but instead of making an empty data set and filling it, you take a copy
    # and reduce it
    def convertToEdited(self, validation_set):
        if self.uses_regression:
            print("Edited Nearest Neighbor is only set up for classification, not regression")
            return

        print("Converting k-Nearest Neighbor to Edited Nearest Neighbor")
        # the training set is going to be modified, so we need a temporary place to hold the training_set
        #original_data = copy.deepcopy(self.training_set)
        #random.shuffle(original_data)
        original_size = len(self.training_set)

        obs_num = 0
        prev_missrate = 1.0
        while (True):
            # move the current observation out of the training set into a temp variable
            #x_i = []
            x_i = copy.deepcopy(self.training_set[obs_num])
            del self.training_set[obs_num]
            #print(len(self.training_set))
            correct_class = x_i[-1]
            #del x_i[-1]
            # classify current observation using the rest of the training set
            predicted_class = self.classify(x_i)
            # if misclassified: add x_i back into training_set
            # if correctly classified: remove x_i
            if predicted_class != correct_class:
                self.training_set.append(x_i)
            else:
                # if a point is removed: start again from the beginning
                obs_num = -1
                # ensure that removing the point doesn't degrade performance
                missrate = self.testClassification(validation_set)
                if missrate > prev_missrate:
                    break
                prev_missrate = missrate
            obs_num += 1
            if obs_num == len(self.training_set):
                break
        print("Original size:",original_size)
        print("Reduced size:",len(self.training_set))

    def getNeighbors(self, new_obs):
        #print("Finding nearest Neighbors")
        # dists: an array of tuples for every item in the training set of the form (training set obs, dist to new obs)
        dists = []
        for x in range(0, len(self.training_set)):
            dist = ms.squaredDistance(new_obs, self.training_set[x], len(self.training_set[0]) - 1)
            # Append the observation from the training set and its distance as a tuple to the distances array
            dists.append((self.training_set[x], dist))
        # sort method uses a key function to be applied to objects to be sorted, so I use this lambda function to tell it
        # to sort by the element at index 1 of the tuple (the distance)
        dists.sort(key=lambda elem: elem[1])
        neighbors = []
        # Now that dists is sorted by distance, the first k elements are the k nearest neighbors
        for x in range(0,self.k):
            # We don't need the distance value anymore, so we only append the training set obs (pull item at index 0
            # from the tuple
            neighbors.append(dists[x][0])
        return neighbors

    def predict(self, new_obs):
        result = 0
        if self.uses_regression:
            result = self.regress(new_obs)
        else:
            result = self.classify(new_obs)
        return result

    # For regression, I chose to use running mean smoother for simplicity.
    def regress(self, new_obs):
        neighbors = self.getNeighbors(new_obs)
        y_bar = 0.0
        for x in range(len(neighbors)):
            y_bar += neighbors[x][-1]
        y_bar /= len(neighbors)
        print("decision from regression:",y_bar)
        return y_bar

    def classify(self, new_obs):
        neighbors = self.getNeighbors(new_obs)
        votes = {}
        for x in range(0, len(neighbors)):
            # Get class
            vote = neighbors[x][len(self.training_set[0])-1]
            # If this class already has votes, add another vote, else add this option to the dictionary
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1
        decision = sorted(votes.items(), key=lambda elem: elem[1], reverse=True)
        print("decision from classification:",decision[0][0])
        return decision[0][0]

    # will return the probability of misclassification
    def testClassification(self, testing_set):
        prob_of_miss = 0
        correct_class = -1
        for obs in range(0, len(testing_set)):
            correct_class = testing_set[obs][-1]
            predicted_class = self.classify(testing_set[obs])
            if predicted_class != correct_class:
                prob_of_miss += 1
        prob_of_miss = float(prob_of_miss) / len(testing_set)
        return prob_of_miss
