import MathAndStats as ms
import math


class GaussianNeuron:

    # assumes each example in examples has class/y
    def __init__(self, mean, examples):
        self.mean = mean
        self.variance = ms.getVariance(mean, examples, len(examples[0]) - 1)


    # calculate the Gaussian output
    # assumes new_example doesn't have class/y
    def getOutput(self, new_example):
        output = 0
        output += -1 * ms.squaredDistance(new_example, self.mean, len(new_example))
        output /= (2.0 * pow(self.variance, 2))
        return math.exp(output)