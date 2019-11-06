import copy
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
    def train(self, input_data, hidden_layer_nodes, eta, alpha, iterations):

        #-create hidden nodes-#
        for layer in range(len(hidden_layer_nodes)):
            inputs = 0
            if layer == 0:
                # if first hidden layer, number of inputs is number of features
                inputs = len(input_data[0])-1
            else:
                # else number of inputs is number of outputs from previous layer
                inputs = len(self.hidden_layers[-1])
            self.hidden_layers.append([])
            for node in range(hidden_layer_nodes[layer]):
                self.hidden_layers[layer].append(unit.Neuron(inputs, self.logistic_nodes))
        #-create output nodes-#
        # set inputs equal to number of features of input data
        inputs = len(input_data[0])-1
        # if there is at least one hidden layer, set number of inputs to the number of nodes in the final layer
        if len(hidden_layer_nodes) != 0:
            inputs = len(self.hidden_layers[-1])
        for output_node in range(self.out_k):
            self.output_layer.append(unit.Neuron(inputs, self.logistic_output))
            if not ((self.output_type == "regression") or (self.output_type == "autoencoder")):
                self.output_layer[output_node].setClass(self.class_list[output_node])

        #-----------------#
        # Backpropogation #
        #-----------------#
        for epoch in range(iterations):
            print(epoch, iterations)
            loss = [0.0] * self.out_k
            for example_num in range(len(input_data)):
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
                            hidden_outputs.append(self.getHiddenLayerOutput(input_data[example_num], 0))
                        else:
                            # else, take outputs from previous layer
                            hidden_outputs.append(self.getHiddenLayerOutput(hidden_outputs[layer-1], layer))
                # there are no hidden layers
                else:
                    hidden_outputs.append(input_data[example_num])
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
                for output_node in range(self.out_k):
                    error = 0
                    if self.output_type == "regression":
                        error = input_data[example_num][-1] - outputs[0][output_node]

                    #else classification
                    prev_error.append(error)
                    loss[output_node] += error * error
                    # shallow copy the weights (alias)
                    weights = self.output_layer[output_node].weights
                    prev_weights.append(copy.deepcopy(weights))
                    for weight_num in range(len(weights)):
                        delta_weight = 0
                        if self.output_type == "regression":
                            delta_weight = (eta*error*hidden_outputs[-1][weight_num])
                    #    #else:
                    #    #    delta_output_weights[output_node].append(eta * error * hidden_outputs[-1][weight_num])
                        weights[weight_num] = weights[weight_num] + delta_weight
                # propogate errors and update weights
                # start at final layer and move backwards
                for layer in range(len(self.hidden_layers)-1, -1, -1):
                    # check if the downstream layer is the outputs
                    if layer == (len(self.hidden_layers)-1):
                        prev_layer_error = prev_error
                    else:
                        # keep error from previous layer
                        prev_layer_error = []
                    current_error = []
                    current_weights = []
                    # iterate through the nodes in a given layer
                    for node in range(len(self.hidden_layers[layer])):
                        sum_downsteam_error = 0.0
                        for downstream_node in range(len(prev_error)):
                            sum_downsteam_error += prev_error[downstream_node] * prev_weights[downstream_node][node+1]
                        activation = hidden_outputs[layer][node+1]
                        current_error.append(sum_downsteam_error * activation * (1-activation))

                        # shallow copy the weights (alias)
                        weights = self.hidden_layers[layer][node].weights
                        current_weights.append(copy.deepcopy(weights))
                        for weight_num in range(len(weights)):
                            delta_weight = 0
                            if self.output_type == "regression":
                                if layer != 0:
                                    delta_weight = (eta * current_error[node] * hidden_outputs[layer-1][weight_num])
                                else:
                                    if weight_num != 0:
                                        delta_weight = (eta * current_error[node] * input_data[example_num][weight_num-1])
                                    else:
                                        delta_weight = eta * current_error[node]
                            weights[weight_num] = weights[weight_num] + delta_weight
                    prev_error = current_error
                    prev_weights = current_weights
            if self.output_type == "regression":
                print("MSE:", (loss[0] / len(input_data)))
            else:
                print("MSE per node:", end=' ')
                # print MSE for each output node
                #for output_num in range(len(self.output_layer)):
                #    print(loss[output_num] / len(data), end=' ')
                print()

    def tune(self, input_data, validation_data, num_layers):
        #train(self, input_data, hidden_layer_nodes, eta, alpha):
        eta = 0.3
        alpha = 0
        hidden_layer_nodes = []
        for layer in range(num_layers):
            less_nodes = 1
            more_nodes = 60
            nodes = random.randint(less_nodes+1, more_nodes-1)
            iterations = 4
            for round in range(iterations):
                less_error = 0
                more_error = 0
                mid_error = 0
                hidden_layer_nodes.append(nodes)
                self.train(input_data, hidden_layer_nodes, 0.3, alpha, 5)
                del hidden_layer_nodes[-1]
                if self.output_type == "regression":
                    # get absolute error for each validation observation
                    results = ms.testRegressor(self, validation_data)
                    for obs in range(len(results)):
                        mid_error += results[obs]
                    # convert to MSE
                    mid_error = (mid_error * mid_error) / len(validation_data)
                self.hidden_layers = []
                self.output_layer = []
                hidden_layer_nodes.append(more_nodes)
                self.train(input_data, hidden_layer_nodes, 0.3, alpha, 5)
                del hidden_layer_nodes[-1]
                if self.output_type == "regression":
                    # get absolute error for each validation observation
                    results = ms.testRegressor(self, validation_data)
                    for obs in range(len(results)):
                        more_error += results[obs]
                    # convert to MSE
                    more_error = (more_error * more_error) / len(validation_data)
                self.hidden_layers = []
                self.output_layer = []
                hidden_layer_nodes.append(less_nodes)
                self.train(input_data, hidden_layer_nodes, 0.3, alpha, 5)
                del hidden_layer_nodes[-1]
                if self.output_type == "regression":
                    # get absolute error for each validation observation
                    results = ms.testRegressor(self, validation_data)
                    for obs in range(len(results)):
                        less_error += results[obs]
                    # convert to MSE
                    less_error = (less_error * less_error) / len(validation_data)
                self.hidden_layers = []
                self.output_layer = []
                if (mid_error <= less_error) and (mid_error <= more_error) and round > 0:
                    break
                elif (more_error <= mid_error) and (more_error <= less_error):
                    old_nodes = nodes
                    nodes = random.randint(nodes+1, more_nodes-1)
                    less_nodes = random.randint(old_nodes+1, nodes-1)
                    more_nodes = random.randint(nodes+1, more_nodes-1)
                elif (less_error <= mid_error) and (less_error <= more_error):
                    old_nodes = nodes
                    nodes = random.randint(less_nodes+1, nodes-1)
                    less_nodes = random.randint(less_nodes+1, nodes-1)
                    more_nodes = random.randint(nodes+1, old_nodes-1)
                else:
                    if more_error <= less_error:
                        old_nodes = nodes
                        nodes = random.randint(nodes + 1, more_nodes - 1)
                        less_nodes = random.randint(old_nodes + 1, nodes - 1)
                        more_nodes = random.randint(nodes + 1, more_nodes - 1)
                    else:
                        old_nodes = nodes
                        nodes = random.randint(less_nodes + 1, nodes - 1)
                        less_nodes = random.randint(less_nodes + 1, nodes - 1)
                        more_nodes = random.randint(nodes + 1, old_nodes - 1)
            hidden_layer_nodes.append(nodes)






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
            hidden_outputs.append(new_obs)
        return self.output_layer[0].getOutput(hidden_outputs)


    def classify(self, new_obs):
        classes = {}
        for output_num in range(len(self.output_layer)):
            hidden_outputs = self.getHiddenOutput(new_obs)
            classes[self.output_layer[output_num].clss] = self.output_layer[output_num].getOutput(hidden_outputs)
        decision = sorted(classes.items(), key=lambda elem: elem[1], reverse=True)
        return decision[0]
