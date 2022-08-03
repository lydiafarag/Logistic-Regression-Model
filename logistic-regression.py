'''
This program has the goal of creating a perceptron that
evaluates non-training inputs trained on the boolean function:
'''

import numpy as np

def unknown_function(A, B, C, D, E, F, G):
  w = A and F and not G
  x = not A and B and C
  y = w or x or D and (not A and F or B and C and G)
  z = y and (E and not F or not B and not C or G and E)
  return int(z)

training_input = [[1,1,1,1,0,1,0],[0,1,0,0,1,0,1],[1,0,0,1,1,0,1],[0,0,0,1,0,0,1],[1,1,1,1,1,1,1],[0,0,1,0,1,0,1],[0,1,1,1,1,0,1],[1,0,0,0,0,0,1],[1,0,1,1,0,0,0],[0,1,1,1,1,1,1],[0,1,1,0,1,1,0],[0,0,0,0,1,1,1],[0,0,1,0,1,1,0],[0,1,1,0,0,1,1],[1,0,0,0,1,1,1],[0,0,0,0,0,0,0]]

training_output = [unknown_function(*x) for x in training_input]

class NeutralNet():

    def __init__(self):
        #   Randomize weights
        self.weights = 2 * np.random.random((len(training_input[0]), 1)) - 1
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            
            #   Compute output using training inputs
            output = self.think(training_inputs)

            #   Error is difference between actual and expected outputs
            error = training_outputs - output

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.weights += adjustments
    
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output
    
if __name__ == "__main__":
    
    neural_net = NeutralNet()
    
    # print("Random starting weights are:")
    # print(neural_net.weights)
    # print("\n")

    training_input = np.array(training_input)

    training_output = np.array([training_output]).T

    neural_net.train(training_input, training_output, 10000)

    # print("Weights after training are:")
    # print(neural_net.weights)
    # print("\n")

    #   Enter test input
    a = int(input("Input A: "))
    b = int(input("Input B: "))
    c = int(input("Input C: "))
    d = int(input("Input D: "))
    e = int(input("Input E: "))
    f = int(input("Input F: "))
    g = int(input("Input G: "))

    #   Compute test output
    print("\n")
    new_data = [a, b, c, d, e, f, g]
    print("New data =", new_data)
    net_out = neural_net.think(np.array(new_data))
    true_out = unknown_function(*new_data)
    print("Anticipated output =", net_out)
    print("Actual output =", true_out)
