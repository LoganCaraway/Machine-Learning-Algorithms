import copy
import time
import MathAndStats as ms
import random
import Neuron as unit


class FeedforwardNetwork:

    # output_type is either "classification", "regression", or "autoencoder". Defaults to classification
    # logistic_nodes and logistic_output are boolean variables determining of the nodes are linear or logistic
    def __init__(self, out_k, clsses, output_type, logistic_nodes, logistic_output):
        self.hidden_layers = []
        self.output_layer = []
        self.output_type = output_type
        self.class_list = clsses
        self.logistic_nodes = logistic_nodes
        self.logistic_output = logistic_output
        # in the case of regression, overwrite these inputs
        if output_type == "regression":
            self.out_k = 1
            logistic_output = False
        else:
            self.out_k = out_k

    def getHiddenLayerOutput(self, new_obs, layer_num):
        data = []
        # bias node
        data.append(1.0)
        for hidden_node_num in range(len(self.hidden_layers[layer_num])):
            data.append(self.hidden_layers[layer_num][hidden_node_num].getOutput(new_obs))
        return data

    # Backpropogation
    def train(self, input_data, hidden_layer_nodes, eta, alpha_momentum, iterations):
        print("Training FF using: eta =",eta,", alpha =",alpha_momentum,", nodes by layer =",hidden_layer_nodes)
        #-create hidden nodes-#
        for layer in range(len(hidden_layer_nodes)):
            if layer == 0:
                # if first hidden layer, number of inputs is number of features-1
                # since the node adds a bias node by default, and that is not desired here
                inputs = len(input_data[0])-2
            else:
                # else number of inputs is number of outputs from previous layer
                inputs = len(self.hidden_layers[-1])
            self.hidden_layers.append([])
            for node in range(hidden_layer_nodes[layer]):
                self.hidden_layers[layer].append(unit.Neuron(inputs, self.logistic_nodes))
        #-create output nodes-#
        # if there is at least one hidden layer, set number of inputs to the number of nodes in the final layer
        if len(hidden_layer_nodes) != 0:
            inputs = len(self.hidden_layers[-1])
        else:
            # set inputs equal to number of features of input data
            inputs = len(input_data[0]) - 2
        for output_node in range(self.out_k):
            self.output_layer.append(unit.Neuron(inputs, self.logistic_output))
            if not ((self.output_type == "regression") or (self.output_type == "autoencoder")):
                self.output_layer[output_node].setClass(self.class_list[output_node])

        #-----------------#
        # Backpropogation #
        #-----------------#
        random.shuffle(input_data)
        prev_delta_weights = []
        prev_loss = []
        worse_epochs = 0
        for epoch in range(iterations):
            loss = [0.0] * self.out_k
            # delta_weights[example_num][layer][node][weight]
            delta_weights = []
            for example_num in range(len(input_data)):
                delta_weights.append([])
                #-------------#
                #-Get Outputs-#
                # -------------#
                # get the output for each hidden node of each hidden layer
                # hidden_outputs[layer][node]
                hidden_outputs = []
                if len(self.hidden_layers) > 0:
                    for layer in range(len(self.hidden_layers)):
                        # if first layer, take data inputs
                        if layer == 0:
                            hidden_outputs.append(self.getHiddenLayerOutput(input_data[example_num][:-1], 0))
                        else:
                            # else, take outputs from previous layer
                            hidden_outputs.append(self.getHiddenLayerOutput(hidden_outputs[-1], layer))
                # there are no hidden layers
                else:
                    hidden_outputs.append(input_data[example_num][:-1])
                # get outputs by layer
                # outputs [layer][node]
                outputs = []
                outputs.append([])
                for output_node in range(self.out_k):
                    outputs[0].append(self.output_layer[output_node].getOutput(hidden_outputs[-1]))
                # -------------#

                #-get output node errors and update output weights-#
                #--------------------------------------------------#
                # prev_error[node]
                prev_error = []
                # prev_weights[node][weight]
                prev_weights = []
                # append a list for output layer
                delta_weights[example_num].append([])
                for output_node in range(self.out_k):
                    # append a list to hold the weights for this node
                    delta_weights[example_num][0].append([])
                    error = 0
                    if self.output_type == "regression":
                        error = input_data[example_num][-1] - outputs[0][output_node]
                    elif self.output_type == "classification":
                        if input_data[example_num][-1] == self.output_layer[output_node].clss:
                            correct = 1
                        else:
                            correct = 0
                        error = correct - outputs[0][output_node]
                        error *= outputs[0][output_node] * (1-outputs[0][output_node])
                    prev_error.append(error)
                    loss[output_node] += error * error
                    # shallow copy the weights (alias)
                    weights = self.output_layer[output_node].weights
                    prev_weights.append(copy.deepcopy(weights))
                    for weight_num in range(len(weights)):
                        delta_weights[example_num][0][output_node].append(eta * error * hidden_outputs[-1][weight_num])
                        if (alpha_momentum > 0) and (example_num > 0):
                            weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][output_node][weight_num] + (alpha_momentum * prev_delta_weights[0][output_node][weight_num])
                        else:
                            weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][output_node][weight_num]
                # propogate errors and update weights
                # start at final layer and move backwards
                prev_layer_error = []
                for layer in range(len(self.hidden_layers)):
                    delta_weights[example_num].append([])
                for layer in range(len(self.hidden_layers)-1, -1, -1):
                    # check if the downstream layer is the outputs
                    if layer == (len(self.hidden_layers)-1):
                        prev_layer_error = prev_error
                    current_error = []
                    current_weights = []
                    # iterate through the nodes in a given layer
                    for node in range(len(self.hidden_layers[layer])):
                        # add list for the weights of this node
                        delta_weights[example_num][layer+1].append([])
                        sum_downsteam_error = 0.0
                        for downstream_node in range(len(prev_layer_error)):
                            sum_downsteam_error += prev_layer_error[downstream_node] * prev_weights[downstream_node][node+1]
                        activation = hidden_outputs[layer][node+1]
                        if self.logistic_nodes:
                            current_error.append(sum_downsteam_error * activation * (1-activation))
                        else:
                            current_error.append(sum_downsteam_error)

                        # shallow copy the weights (alias)
                        weights = self.hidden_layers[layer][node].weights
                        current_weights.append(copy.deepcopy(weights))
                        for weight_num in range(len(weights)):
                            if layer != 0:
                                delta_weights[example_num][layer+1][node].append(eta * current_error[node] * (hidden_outputs[layer-1][weight_num]))
                            else:
                                delta_weights[example_num][layer+1][node].append(eta * current_error[node] * (input_data[example_num][weight_num]))
                            if (alpha_momentum > 0) and (example_num > 0):
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][layer+1][node][weight_num] + alpha_momentum * prev_delta_weights[layer+1][node][weight_num]
                            else:
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][layer+1][node][weight_num]
                    prev_layer_error = current_error
                    prev_weights = current_weights
                if alpha_momentum > 0:
                    prev_delta_weights = delta_weights[-1]
            if self.output_type == "regression":
                print("FF:",epoch,"of",iterations, end=' ')
                print("MSE:", (loss[0] / len(input_data)))
            else:
                print("FF:",epoch,"of",iterations, end=' ')
                print("MSE per node:", end=' ')
                # print MSE for each output node
                for output_num in range(len(self.output_layer)):
                    print(loss[output_num] / len(input_data), end=' ')
                print()

            if epoch > 10:
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

    def tune(self, input_data, validation_data, num_layers):
        eta = 0.05
        alpha = 0
        hidden_layer_nodes = []
        #self.train(input_data, hidden_layer_nodes, eta, alpha, 10)
        print("Tuning FF nodes per layer for",num_layers,"layers")
        for layer in range(num_layers):
            less_nodes = 1
            more_nodes = 100
            nodes = random.randint(less_nodes+1, more_nodes-1)
            iterations = 4
            for round in range(iterations):
                less_error = 0
                more_error = 0
                mid_error = 0
                hidden_layer_nodes.append(nodes)
                self.train(input_data, hidden_layer_nodes, eta, alpha, 10)
                del hidden_layer_nodes[-1]
                if self.output_type == "regression":
                    # get absolute error for test
                    results = ms.testRegressor(self, validation_data)
                    for obs in range(len(results)):
                        mid_error += (results[obs] * results[obs])
                    mid_error /= len(results)
                elif self.output_type == "classification":
                    mid_error = self.testClassification(validation_data)
                self.hidden_layers = []
                self.output_layer = []
                hidden_layer_nodes.append(more_nodes)
                self.train(input_data, hidden_layer_nodes, eta, alpha, 10)
                del hidden_layer_nodes[-1]
                if self.output_type == "regression":
                    # get absolute error for test
                    results = ms.testRegressor(self, validation_data)
                    for obs in range(len(results)):
                        more_error += (results[obs] * results[obs])
                    more_error /= len(results)
                elif self.output_type == "classification":
                    more_error = self.testClassification(validation_data)
                self.hidden_layers = []
                self.output_layer = []
                hidden_layer_nodes.append(less_nodes)
                self.train(input_data, hidden_layer_nodes, eta, alpha, 10)
                del hidden_layer_nodes[-1]
                if self.output_type == "regression":
                    # get error for test
                    results = ms.testRegressor(self, validation_data)
                    for obs in range(len(results)):
                        less_error += (results[obs] * results[obs])
                    less_error /= len(results)
                elif self.output_type == "classification":
                    less_error = self.testClassification(validation_data)
                self.hidden_layers = []
                self.output_layer = []
                if (mid_error <= less_error) and (mid_error <= more_error) and round > 0:
                    break
                elif (more_error <= mid_error) and (more_error <= less_error):
                    old_nodes = nodes
                    try:
                        nodes = random.randint(nodes+1, more_nodes-1)
                        less_nodes = random.randint(old_nodes+1, nodes-1)
                        more_nodes = random.randint(nodes+1, more_nodes-1)
                    except:
                        nodes = old_nodes
                        break
                elif (less_error <= mid_error) and (less_error <= more_error):
                    old_nodes = nodes
                    try:
                        nodes = random.randint(less_nodes+1, nodes-1)
                        less_nodes = random.randint(less_nodes+1, nodes-1)
                        more_nodes = random.randint(nodes+1, old_nodes-1)
                    except:
                        nodes = old_nodes
                        break
                else:
                    if more_error <= less_error:
                        old_nodes = nodes
                        try:
                            nodes = random.randint(nodes + 1, more_nodes - 1)
                            less_nodes = random.randint(old_nodes + 1, nodes - 1)
                            more_nodes = random.randint(nodes + 1, more_nodes - 1)
                        except:
                            nodes = old_nodes
                            break
                    else:
                        old_nodes = nodes
                        try:
                            nodes = random.randint(less_nodes + 1, nodes - 1)
                            less_nodes = random.randint(less_nodes + 1, nodes - 1)
                            more_nodes = random.randint(nodes + 1, old_nodes - 1)
                        except:
                            nodes = old_nodes
                            break
            hidden_layer_nodes.append(nodes)
        print("Selected nodes by layer: ",hidden_layer_nodes)
        lowest_eta = -1
        lowest_error = -1
        print("Tuning FF eta")
        while eta <= 0.5:
            self.train(input_data, hidden_layer_nodes, eta, alpha, 10)
            error = 0
            if self.output_type == "regression":
                # get error for test
                results = ms.testRegressor(self, validation_data)
                for obs in range(len(results)):
                    error += (results[obs] * results[obs])
                error /= len(results)
            elif self.output_type == "classification":
                error = self.testClassification(validation_data)
            print("MSE for eta =", eta, ":", error, "lowest MSE =", lowest_error)
            if (error < lowest_error) or (eta == 0.05):
                lowest_eta = eta
                lowest_error = error
            eta += 0.05
            self.hidden_layers = []
            self.output_layer = []
        print("Selected FF eta =", lowest_eta)

        print("Tuning FF alpha for momentum")
        lowest_alpha = 0
        lowest_error = -1
        while alpha < 0.5:
            self.train(input_data, hidden_layer_nodes, lowest_eta, alpha, 10)
            error = 0
            if self.output_type == "regression":
                # get error for test
                results = ms.testRegressor(self, validation_data)
                for obs in range(len(results)):
                    error += (results[obs] * results[obs])
                error /= len(results)
            elif self.output_type == "classification":
                error = self.testClassification(validation_data)
            print("MSE for alpha =", alpha, ":", error, "lowest MSE =", lowest_error)
            if (error < lowest_error) or (alpha == 0):
                lowest_alpha = alpha
                lowest_error = error
            prev_error = error
            alpha += 0.1
            self.hidden_layers = []
            self.output_layer = []
        print("Selected FF alpha =", lowest_alpha)
        now = time.time()
        self.train(input_data, hidden_layer_nodes, lowest_eta, lowest_alpha, 100)
        done = time.time()
        self.convergence_time = done - now



    def predict(self, new_obs):
        if self.output_type == "regression":
            return self.regress(new_obs)
        else:
            # I have classify return (class, probability) as a tuple for use in tuning, but
            # predict will simply return class
            return self.classify(new_obs)[0]

    def regress(self, new_obs):
        # get the output for each hidden node of each hidden layer
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        return self.output_layer[0].getOutput(hidden_outputs)
        # hidden_outputs[layer][node]
        #hidden_outputs = []
        #if len(self.hidden_layers) > 0:
        #    for layer in range(len(self.hidden_layers)):
        #        # if first layer, take data inputs
        #        if layer == 0:
        #            hidden_outputs.append(self.getHiddenLayerOutput(new_obs, 0))
        #        else:
        #            # else, take outputs from previous layer
        #            hidden_outputs.append(self.getHiddenLayerOutput(hidden_outputs[-1], layer))
        ## there are no hidden layers
        #else:
        #    hidden_outputs.append(new_obs)
        #return self.output_layer[0].getOutput(hidden_outputs[-1])


    def classify(self, new_obs):
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        #return self.output_layer[0].getOutput(hidden_outputs)
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
            predicted = self.classify(testing_set[obs][:-1])
            if predicted[0] == correct_class:
                error = (1 - predicted[1])
            else:
                error = predicted[1]
            mse += error * error
        return mse / len(testing_set)