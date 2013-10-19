# NeuralNet.py
# A Python implementation of a neural network to distinguish males
# from females. Hoping to get better than that lousy 50% accuracy
# that the Java version had. Using Python cause it's easy and nice.

# This version handles multiple hidden layers, but it doesn't work and
# is a pain, so I'm switching to a version that only allows one
# hidden layer unless I need more of them.
import argparse
import math
import pdb
import numpy as np
from numpy import shape

global debug

class Network:
    def __init__(self, numnodes, learningrate=0.02, initInterval=0.1, debug=False):
    	"""
    	numnodes is a list of the number of nodes in each layer, including the input layer 
    	and the output layer.
    	"""
        def debug_weights():
            determined_weights = [np.array([-0.04693253, -0.04471532, -0.01004506]),
                                  np.array([-0.02162086,  0.00055035, -0.01560475]),
                                  np.array([ 0.02513274, -0.02008302,  0.01826699]),
                                  np.array([ 0.04492586,  0.00102442,  0.00779095]),
                                  np.array([-0.01094794, -0.04528093, -0.01694642])]
            for w in determined_weights:
                yield w
        #end debug_weights

        self.numinputs = numnodes[0]
        self.learningrate = learningrate
        if not debug:
            self.layers = [Layer(numnodes[i-1], numnodes[i], initInterval) for i in range(1, len(numnodes))]
        else:
            weights = debug_weights()
            self.layers = [Layer(numnodes[i-1], numnodes[i], next(weights)) for i in range(1, len(numnodes))]
    
    def feed_network(self, inputs):
        """
	inputs is a numpy array of input values. Each index of
        the input value must correspond to one input neuron.
	"""
        assert (len(inputs) == self.numinputs)
        currentin = inputs
        for layer in self.layers:
            #if debug: print(layer.__repr__())
            currentin = layer.calc_outputs(currentin)
        self.output = currentin
        return self.output

    def propagate_back(self, example, answer):
        """
        example is the data to give to feed_network
        answer is a numpy array of the correct outputs for this example.
        """
        self.feed_network(example)
        pdb.set_trace()
        outvals = self.layers[-1].output
        gprime = outvals * (np.ones(len(outvals)) - outvals)
        delta = gprime * (answer - outvals)
        # Calculate delta vectors for all weight layers.
        delta = self.layers[-1].calc_delta(delta, output=True)
        # You skipped one. Remember, the delta we calculate here needs
        # to update the weight matrix stored inside the output layer.
        # But we're skipping over it because its delta needs to be
        # calculated differently from the others.
        for layer in reversed(self.layers[:-1]):
            delta = layer.calc_delta(delta, output=False)
        #print("Output delta: {0}".format(self.layers[-1].delta))
        for layer in self.layers:
            layer.update_weights(self.learningrate)
    
    def judgment_on(self, data, decision):
        """
        data is an input vector for which we want a decision.
        decision is a function of one argument, a list whose length is
        the number of outputs of the network, that decides how to classify
        data based on what the network outputs for it. (e.g. the 0 above .5/1 below
        decision, we could pass in lambda out: 1 if out[0] > 0.5 else 0.)
        """
        return decision(data)
    
    def print_me(self):
        for layer in self.layers:
            print(layer)
        
#end Network

class Layer:
    def __init__(self, numInputs, numNodes, initInterval):
    	"""
    	numInputs is the number of nodes in the previous layer, each of which
    	contributes an input.
    	numNodes is the number of nodes in this layer. We need a numInputs x numNodes
    	matrix to represent the weights of all this layer's nodes assigned to the inputs
    	from the previous layer. weights[i,j] is the weight given the ith input by the jth
    	neuron in this layer.
    	"""
        self.weights = np.random.uniform(-initInterval, initInterval, (numInputs, numNodes))
        self.numInputs = numInputs
        self.numNodes = numNodes
        #if debug: print(self.weights)

    def calc_outputs(self, inputs):
    	"""
    	self.output[i] is the output of the ith node in this layer
    	"""
    	assert len(self.weights) == len(inputs), "Layer.calc_outputs: mismatched inputs."
        self.inputs = inputs
    	self.weightedsums = np.dot(inputs, self.weights)
        self.output = np.array(map(lambda x : 1/(1 + math.exp(-x)), self.weightedsums))
        # Note: might need to add a w_0 weight.
        return self.output  
        # Both save and return for debugging purposes (to see the
        # dimensionality)

    def calc_delta(self, previousDelta, output=False):
        """
        previousDelta is the delta array from the previous layer.
        """
        #print("weights is ", self.weights)
        #print(previousDelta)
        #print("calc_delta:\n\tpreviousDelta's shape is {0:<10} weights's shape is {0:<10}".format(
        #    np.shape(previousDelta), np.shape(self.weights)))
        gprime =  self.output * (np.ones(len(self.output)) - self.output)
        prevcontrib = np.dot(self.weights, previousDelta)
        #print("gprime: {0:<10}\nprevcontrib{1:<10}".format(np.shape(gprime), np.shape(prevcontrib)))
        if not output:
            self.delta = gprime * prevcontrib
            return self.delta
        else:
            self.delta = previousDelta
            return gprime*prevcontrib

    def update_weights(self, learningrate):
        u = learningrate * np.outer(self.inputs, self.delta)
        #print("{0:<10} is dim of u {1:>10} is dim of delta {2:>10} is dim of weights".format(np.shape(u), 
        #                                                                                     np.shape(self.delta),
        #                                                                                     np.shape(self.weights)))
        assert np.shape(u) == np.shape(self.weights)
        self.weights = self.weights + u

    def __repr__(self):
    	return "Layer(inputs={0}, nodes={1}, weights={2})".format(self.numInputs, self.numNodes, self.weights)
# end Layer

if __name__ == "__main__":
    args = argparse.ArgumentParser(description=
                                   "Takes train, test, and verify options.")
    args.add_argument("-train", nargs=2)
    args.add_argument("-test", action='store')
    args.add_argument("-verify", action='store_true')
    args.add_argument("-debug", action='store_true')
    argnames = args.parse_args()
    debug = argnames.debug
    
    if debug:
    	print(argnames)
    	l = Layer(3, 3)
    	print(l.calc_outputs(np.array([1, 2, 3])))
    	n = Network([3, 3, 1])
    	print(n.layers)
    	print(n.feed_network(np.array([0, 1, 1])))

 
    # When it's time to read in data, do it like this. But with functions.
    #fname = os.path.join(argvals.train[0], os.listdir(argvals.train[0])[0])
    #indata = np.fromfile(fname, dtype=int, count=-1, sep=' ')
    #print(indata, " ", indata.shape)
    
# Some things to do if it sucks:
# SVD the picture data to make things sleeker
# Implement some algorithm to eliminate certain links if they might be creating
# too much extra noise. (Give the links you want to get rid of a weight of zero.)
# Lower the learning rate
# Raise the number of hidden units--according to AIMA (page 732), there is a proof that
# the required number of hidden units to represent a function with a neural network grows
# exponentially in the number of inputs. E.g. 2^n / n hidden units are needed to encode a Boolean
# function of n inputs. Before we had like 30. We have 1024 inputs. No wonder it couldn't get
# above 50% accuracy. If an input can have 128 values (seems like the pixels use 8-bit greyscale)
# and there are 1024 inputs, we'd need about 128^1024 / 1024 =~ 5.93e2154 hidden units.

#Note: this works:
# a = [1, 2, 3]
# b = [5, 6, 7]
# c = [9, 10, 11]
# map(lambda x, y, z: x+y+z, a, b, c)
