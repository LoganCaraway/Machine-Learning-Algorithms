import copy
import random
import time
import MathAndStats as ms
import GaussianNeuron as gaussian
import Neuron as unit

class RBFNetwork:

    def __init__(self, means, clusts, clsses, uses_regression, logistic_output):
        self.hidden_layer = []
        self.output_layer = []
        self.uses_regression = uses_regression
        # in the case of regression, overwrite these inputs
        if uses_regression:
            self.out_k = 1
            logistic_output = False
        else:
            self.out_k = len(clsses)
        # add hidden Gaussians
        for hidden_node in range(len(means)):
            self.hidden_layer.append(gaussian.GaussianNeuron(means[hidden_node], clusts[hidden_node]))
        # add output node(s)
        for output_node in range(self.out_k):
            self.output_layer.append(unit.Neuron(len(means), logistic_output))
            if not uses_regression:
                self.output_layer[output_node].setClass(clsses[output_node])

    def getHiddenOutput(self, new_obs):
        data = []
        # bias node
        data.append(1.0)
        for hidden_node_num in range(len(self.hidden_layer)):
            data.append(self.hidden_layer[hidden_node_num].getOutput(new_obs))
        return data

    # Gradient Descent
    def trainOutputLayer(self, input_data, eta, alpha_momentum, iterations):
        print("Training RBF using: eta =", eta, ", alpha =", alpha_momentum)
        random.shuffle(input_data)
        data = []
        for example_num in range(len(input_data)):
            data.append(self.getHiddenOutput(input_data[example_num][:-1]))
        prev_delta_weights = []
        prev_loss = []
        worse_epochs = 0
        for epoch in range(iterations):
            loss = [0.0] * self.out_k
            # delta_weights[example_num][node][weight]
            delta_weights = []
            #weights_for_examples = []
            for example_num in range(len(data)):
                delta_weights.append([])
                if self.uses_regression:
                    # append a list to hold the weights for this layer
                    delta_weights[example_num].append([])
                    predicted = self.output_layer[0].getOutput(data[example_num])
                    error = input_data[example_num][-1] - predicted
                    squared_loss = error * error
                    loss[0] += squared_loss
                    # shallow copy the weights (alias)
                    weights = self.output_layer[0].weights
                    # update weights
                    for weight_num in range(len(weights)):
                        delta_weights[example_num][0].append(eta*error*data[example_num][weight_num])
                        if (alpha_momentum > 0) and (example_num > 0):
                            weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][weight_num] + (alpha_momentum * prev_delta_weights[0][weight_num])
                        else:
                            weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][weight_num]
                else:
                    weights_for_output_node = []
                    for output_num in range(len(self.output_layer)):
                        # append a list to hold the weights for this layer
                        delta_weights[example_num].append([])
                        # positive class is the one held in the one-hot node
                        # negative class is any class not held in the one-hot node
                        prob = self.output_layer[output_num].getOutput(data[example_num])
                        if input_data[example_num][-1] == self.output_layer[output_num].clss:
                            correct = 1
                        else:
                            correct = 0
                        error = correct - prob
                        # shallow copy the weights (alias)
                        weights = self.output_layer[output_num].weights
                        # update weights
                        if error != 0:
                            for weight_num in range(len(weights)):
                                delta_weights[example_num][output_num].append(eta * error * prob * (1.0 - prob) * data[example_num][weight_num])
                                if (alpha_momentum > 0) and (example_num > 0):
                                    weights[weight_num] = weights[weight_num] + delta_weights[example_num][output_num][weight_num] + (alpha_momentum * prev_delta_weights[output_num][weight_num])
                                else:
                                    weights[weight_num] = weights[weight_num] + delta_weights[example_num][output_num][weight_num]
                        squared_loss = error * error
                        loss[output_num] += squared_loss
                if alpha_momentum > 0:
                    prev_delta_weights = delta_weights[-1]
            if self.uses_regression:
                print("RBF:", epoch,"of",iterations, end=' ')
                print("MSE:", (loss[0] / len(data)))
            else:
                print("RBF:",epoch,"of",iterations, end=' ')
                print("MSE per node:", end=' ')
                # print MSE for each output node
                for output_num in range(len(self.output_layer)):
                    print(loss[output_num] / len(data), end=' ')
                print()

            if epoch > 0:
                better = True
                for output_num in range(len(self.output_layer)):
                    if loss[output_num] < prev_loss[output_num]:
                        better = False
                        break
                if better:
                    worse_epochs += 1
                    if worse_epochs > 2:
                        print("Converged")
                        break
                else:
                    worse_epochs = 0
            prev_loss = loss

    def tune(self, input_data, validation_data):
        print("Tuning RBF Network")
        eta = 0.05
        lowest_eta = -1
        lowest_error = -1
        print("Tuning RBF eta")
        while eta <= 0.5:
            self.trainOutputLayer(input_data, eta, 0, 10)
            error = 0
            if self.uses_regression:
                # get absolute error for test
                results = ms.testRegressor(self, validation_data)
                for obs in range(len(results)):
                    error += (results[obs] * results[obs])
                error /= len(results)
            else:
                error = self.testClassification(validation_data)
            print("MSE for eta =",eta,":",error,"lowest MSE =",lowest_error)
            if (error < lowest_error) or (eta == 0.05):
                lowest_eta = eta
                lowest_error = error
            eta += 0.05
            for node in range(len(self.output_layer)):
                self.output_layer[node].resetWeights()
        print("Selected RBF eta =",lowest_eta)
        #self.trainOutputLayer(input_data, eta, 0)

        print("Tuning RBF alpha for momentum")
        alpha = 0
        lowest_alpha = 0
        lowest_error = -1
        while alpha < 0.5:
            self.trainOutputLayer(input_data, lowest_eta, alpha, 10)
            error = 0
            if self.uses_regression:
                # get absolute error for each validation observation
                results = ms.testRegressor(self, validation_data)
                for obs in range(len(results)):
                    error += (results[obs] * results[obs])
                error /= len(results)
            else:
                error = self.testClassification(validation_data)
            print("MSE for alpha =",alpha,":",error,"lowest MSE =",lowest_error)
            if (error < lowest_error) or (alpha == 0):
                lowest_alpha = alpha
                lowest_error = error
            prev_error = error
            alpha += 0.1
            for node in range(len(self.output_layer)):
                self.output_layer[node].resetWeights()
        print("Selected RBF alpha =", lowest_alpha)
        now = time.time()
        self.trainOutputLayer(input_data, lowest_eta, lowest_alpha, 100)
        done = time.time()
        self.convergence_time = done - now


    # predict the value for a new observation
    def predict(self, new_obs):
        if self.uses_regression:
            return self.regress(new_obs)
        else:
            # I have classify return (class, probability) as a tuple for use in tuning, but
            # predict will simply return class
            return self.classify(new_obs)[0]

    def regress(self, new_obs):
        hidden_outputs = self.getHiddenOutput(new_obs)
        return self.output_layer[0].getOutput(hidden_outputs)


    def classify(self, new_obs):
        hidden_outputs = self.getHiddenOutput(new_obs)
        classes = {}
        for output_num in range(len(self.output_layer)):
            classes[self.output_layer[output_num].clss] = self.output_layer[output_num].getOutput(hidden_outputs)
        decision = sorted(classes.items(), key=lambda elem: elem[1], reverse=True)
        return decision[0]


    # will return the MSE for classification
    def testClassification(self, testing_set):
        mse = 0
        for obs in range(len(testing_set)):
            correct_class = testing_set[obs][-1]
            predicted = self.classify(testing_set[obs])
            if predicted[0] == correct_class:
                error = (1 - predicted[1])
            else:
                error = predicted[1]
            mse += error * error
        return mse / len(testing_set)