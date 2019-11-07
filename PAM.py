import MathAndStats as ms
import copy
import random


class PAM:

    def __init__(self, data, k, uses_regression, min_examples_in_cluster):
        print("Finding medoids")
        # appended to using deep copy, holds the medoids
        self.medoids = []
        # appended to using shallow copy, holds the items clustered around a given medoid
        self.clust = []
        # randomize original data
        random.shuffle(data)
        # randomize initial medoids
        for i in range(k-1, -1, -1):
            # randomly initialize medoids to values in the data set
            self.medoids.append(copy.deepcopy(data[i]))
            del data[i]
            # initialize clusters to empty arrays
            self.clust.append([])
        previous_medoids = []
        loop_num = 0
        # PAM is an iterative approach meaning that even if it isn't run to convergence,
        # it is still fairly accurate. Each iteration gives better and better medoids, but
        # just like an infinite series: you only need some iterations to get a good approximation.
        # Due to this, PAM is infrequently ran to convergence, it is typically stopped after a
        # certain number of iterations where more iterations will give slightly more accurate
        # results. For the given data sets, I found 6 iterations to be sufficient to give good
        # estimates for the medoids.
        while self.medoidsMoved(previous_medoids) and (loop_num < 3):
            print("Loop", loop_num)
            #previous_medoids = copy.deepcopy(self.medoids)
            previous_medoids = self.medoids
            for x_i in range(len(data)):
                # start with considering the first medoid the closest
                closest_medoid = self.medoids[0]
                closest_distance = ms.squaredDistance(closest_medoid, data[x_i], len(closest_medoid)-1)
                # find argmin m_j ( distance(x_i, m_j) )
                for m_j in range(1, k):
                    temp_dist = ms.squaredDistance(self.medoids[m_j], data[x_i], len(self.medoids[m_j])-1)
                    if temp_dist < closest_distance:
                        closest_distance = temp_dist
                        closest_medoid = self.medoids[m_j]
                # assign x_i to medoid m_j
                self.clust[self.medoids.index(closest_medoid)].append(data[x_i])
            # calculate distortion
            distor = self.distortion()
            # for each m_i in medoids do
            for m_i in range(k):
                # for each example x_j in data where x_j is not in m_i
                for x_j in range(len(data)):
                    if data[x_j] not in self.clust[m_i]:
                        # swap m_i and x_j
                        temp_ex = copy.deepcopy(data[x_j])
                        data[x_j] = copy.deepcopy(self.medoids[m_i])
                        self.medoids[m_i] = temp_ex
                        distor_prime = self.distortion()
                        # swap back
                        if distor <= distor_prime:
                            temp_ex = copy.deepcopy(data[x_j])
                            data[x_j] = copy.deepcopy(self.medoids[m_i])
                            self.medoids[m_i] = temp_ex
            loop_num += 1
        # repeat until no change in medoids (assumming running until convergence) or the specified loop number is reached
        for medoid_num in range(len(self.medoids) - 1, -1, -1):
            if len(self.clust[medoid_num]) < min_examples_in_cluster:
                del self.clust[medoid_num]
                del self.medoids[medoid_num]
                continue
            # add a medoid to its cluster since the medoid is an observation
            self.clust[medoid_num].append(self.medoids[medoid_num])
            if uses_regression:
                #self.medoids[medoids_num].append(self.regress(medoids_num))
                self.medoids[medoid_num][-1] = self.regress(medoid_num)
            else:
                #self.medoids[medoids_num].append(self.classify(medoids_num))
                self.medoids[medoid_num][-1] = self.classify(medoid_num)

    # check whether the medoids moved
    def medoidsMoved(self, old_medoids):
        # different lengths mean they are not the same
        if len(self.medoids) != len(old_medoids):
            return True
        for medoid_num in range(len(self.medoids)):
            for feature_num in range(len(self.medoids[0])):
                if self.medoids[medoid_num][feature_num] != old_medoids[medoid_num][feature_num]:
                    return True
        return False

    def distortion(self):
        distortion = 0
        # for each medoid j
        for j in range(len(self.medoids)):
            # for each example i owned by cluster j
            for i in range(len(self.clust[j])):
                # for each feature f
                example_distance = 0
                for f in range(len(self.clust[j][0]) -1):
                    # find the absolute distance for each feature
                    example_distance += abs(self.clust[j][i][f] - self.medoids[j][f])
                # take the square of the absolute distance for the squared distance
                distortion += pow(example_distance, 2)
        return distortion


    def predict(self, new_obs):
        # dists: an array of tuples for every item in the training set of the form (training set obs, dist to new obs)
        dists = []
        for x in range(len(self.medoids)):
            dist = ms.squaredDistance(new_obs, self.medoids[x], len(self.medoids[0]) - 1)
            # Append the observation from the medoid list and its distance as a tuple to the distances array
            dists.append((self.medoids[x], dist))
        # sort method uses a key function to be applied to objects to be sorted, so I use this lambda function to tell it
        # to sort by the element at index 1 of the tuple (the distance)
        dists.sort(key=lambda elem: elem[1])
        # [closest tuple][first elem of tuple (medoid)][last elem of medoid]
        return dists[0][0][-1]

    # For regression, I chose to use running mean smoother for simplicity.
    def regress(self, medoid_num):
        neighbors = self.clust[medoid_num]
        y_bar = 0.0
        for x in range(len(neighbors)):
            y_bar += neighbors[x][-1]
        y_bar /= len(neighbors)
        print("assigning", y_bar, "to medoid from regression")
        return y_bar

    def classify(self, medoid_num):
        neighbors = self.clust[medoid_num]
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
        print("assigning",decision[0][0],"to medoid from classification")
        return decision[0][0]