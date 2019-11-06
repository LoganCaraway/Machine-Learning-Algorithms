import MathAndStats as ms
import copy
import random


class KMeans:

    def __init__(self, data, k, uses_regression, min_examples_in_cluster):
        print("Finding centroids")
        self.centroids = [[]] * k
        self.clust = []
        # randomize original data
        random.shuffle(data)
        for i in range(k-1, -1, -1):
            # randomly initialize centroids to values in the data set (not required to be values in the data set, this is supposed to make it converge faster)
            self.centroids[i] = copy.deepcopy(data[i])
            del self.centroids[i][-1]
        oldState = [] #We first define an oldState variable to remember the place the centroids were in
        while (self.centroidsMoved(oldState)):
            oldState = copy.deepcopy(self.centroids)  #if we get to the start of the loop, we'll need to deep copy the last state of Centroids for later
            #We reset the Clusters 2 steps, a clear, then we fill it with k empty indices
            self.clust = []
            for i in range(k):
                self.clust.append([])
            for xi in range(len(data)):
                #Clustering starts here. Let's split each observation where it needs to go.
                closestCentroid = self.centroids[0] #We autodefine the first centroid as the closest point for now
                closestDistance = ms.squaredDistance(self.centroids[0],data[xi],len(self.centroids[0]))
                for y in range(1,len(self.centroids)): #We now look for actual the Argmin
                    tempDistance = ms.squaredDistance(self.centroids[y],data[xi],len(self.centroids[y]))
                    if tempDistance < closestDistance:
                        closestDistance = tempDistance
                        closestCentroid = self.centroids[y]
                #At the end of the loop, when the closest Centroid has been defined, stick the data line to the closest Cluster
                self.clust[self.centroids.index(closestCentroid)].append(copy.deepcopy(data[xi]))
            #Recalculating Centroids now.
            # reset centroids to 0
            for centroid_num in range(len(self.centroids)):
                for feature_num in range(len(self.centroids[0])):
                    self.centroids[centroid_num][feature_num] = 0
            # sum elements from clusters into centroids and divide: centroids will be the average coordinates of obs in their cluster
            for centroid_num in range(len(self.centroids)):
                for clust_obs in range(len(self.clust[centroid_num])):
                    for feature_num in range(len(self.centroids[0])):
                        self.centroids[centroid_num][feature_num] += self.clust[centroid_num][clust_obs][feature_num]
                for feature_num in range(len(self.centroids[0])):
                    if len(self.clust[centroid_num]) != 0:
                        self.centroids[centroid_num][feature_num] /= len(self.clust[centroid_num])

        for centroid_num in range(len(self.centroids)-1, -1, -1):
            if (len(self.clust[centroid_num]) < min_examples_in_cluster) or (len(self.clust[centroid_num]) == 0):
                del self.clust[centroid_num]
                del self.centroids[centroid_num]
                continue
            if uses_regression:
                self.centroids[centroid_num].append(self.regress(centroid_num))
            else:
                self.centroids[centroid_num].append(self.classify(centroid_num))

    # check whether the centroids moved
    def centroidsMoved(self, old_centroids):
        # different lengths mean they are not the same
        if len(self.centroids) != len(old_centroids):
            return True
        for centroid_num in range(len(self.centroids)):
            for feature_num in range(len(self.centroids[0])):
                if self.centroids[centroid_num][feature_num] != old_centroids[centroid_num][feature_num]:
                    return True
        return False

    def predict(self, new_obs):
        # dists: an array of tuples for every item in the training set of the form (training set obs, dist to new obs)
        dists = []
        for x in range(len(self.centroids)):
            dist = ms.squaredDistance(new_obs, self.centroids[x], len(self.centroids[0]) - 1)
            # Append the observation from the centroid list and its distance as a tuple to the distances array
            dists.append((self.centroids[x], dist))
        # sort method uses a key function to be applied to objects to be sorted, so I use this lambda function to tell it
        # to sort by the element at index 1 of the tuple (the distance)
        dists.sort(key=lambda elem: elem[1])
        # [closest tuple][first elem of tuple (centroid)][last elem of centroid]
        return dists[0][0][-1]

    # For regression, I chose to use running mean smoother for simplicity.
    def regress(self, centroid_num):
        neighbors = self.clust[centroid_num]
        y_bar = 0.0
        for x in range(len(neighbors)):
            y_bar += neighbors[x][-1]
        y_bar /= len(neighbors)
        print("assigning", y_bar, "to centroid from regression")
        return y_bar

    def classify(self, centroid_num):
        neighbors = self.clust[centroid_num]
        votes = {}
        for x in range(len(neighbors)):
            # Get class
            vote = neighbors[x][-1]
            # If this class already has votes, add another vote, else add this option to the dictionary
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1
        decision = sorted(votes.items(), key=lambda elem: elem[1], reverse=True)
        print("assigning",decision[0][0],"to centroid from classification")
        return decision[0][0]