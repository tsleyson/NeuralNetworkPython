# NeuralNet.py
# A Python implementation of a neural network to distinguish males
# from females. Hoping to get better than that lousy 50% accuracy
# that the Java version had. Using Python cause it's easy and nice.
import argparse
import math
import pdb
import numpy as np
from numpy import shape

global debug

class Network:
    def __init__(self, 
                 numnodes, 
                 learningrate=0.02, 
                 initInterval=0.1, 
                 activation=lambda x : 1/(1+math.exp(-x)), 
                 derivative=lambda outvals : outvals * (1 - outvals),
                 debug=False):
    	"""
    	numnodes is a list of the number of nodes in each layer, including the input layer 
    	and the output layer.
        initInterval is the size of the interval around zero from which the random starting
        weights are drawn. 
    	"""
        self.numinputs = numnodes[0]
        self.learningrate = learningrate
        self.initInterval = initInterval
        self.derivative = derivative
        if not debug:
            self.layers = [Layer(numnodes[i-1], numnodes[i], 
                                 initInterval, activation) for i in range(1, len(numnodes))]
        # See the abandoned multihidden code for a cool iterator trick
        # to set predetermined weight matrices.

    def feed_network(self, inputs):
        """
	inputs is a numpy array of input values. Each index of
        the input value must correspond to one input neuron.
	"""
        try:
            assert len(inputs) == self.numinputs
        except:
            pdb.set_trace()
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
        #pdb.set_trace()
        outputlayer = self.layers[-1]
        hiddenlayer = self.layers[-2]

        outputlayer.delta = self.derivative(outputlayer.output) * (answer - outputlayer.output)
        
        err = np.dot(outputlayer.weights, outputlayer.delta)
        hiddenlayer.delta = self.derivative(hiddenlayer.output) * err
        
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

    def __repr__(self):
        return "Network(numinputs={0}, learningrate={1}, initInterval={2})".format(self.numinputs,
                                                                                   self.learningrate,
                                                                                   self.initInterval)
#end Network

class Layer:
    def __init__(self, numInputs, numNodes, initInterval, activation):
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
        self.activation = activation
        #if debug: print(self.weights)

    def calc_outputs(self, inputs):
    	"""
    	self.output[i] is the output of the ith node in this layer
    	"""
    	assert len(self.weights) == len(inputs), "Layer.calc_outputs: mismatched inputs."
        self.inputs = inputs
    	self.weightedsums = np.dot(inputs, self.weights)
        self.output = np.array(map(self.activation, self.weightedsums))
        # Note: might need to add a w_0 weight.
        return self.output  
        # Both save and return for debugging purposes (to see the
        # dimensionality)

    # def calc_delta(self, previousDelta, output=False):
    #     """
    #     previousDelta is the delta array from the previous layer.
    #     """
    #     #print("weights is ", self.weights)
    #     #print(previousDelta)
    #     #print("calc_delta:\n\tpreviousDelta's shape is {0:<10} weights's shape is {0:<10}".format(
    #     #    np.shape(previousDelta), np.shape(self.weights)))
    #     gprime =  self.output * (np.ones(len(self.output)) - self.output)
    #     prevcontrib = np.dot(self.weights, previousDelta)
    #     #print("gprime: {0:<10}\nprevcontrib{1:<10}".format(np.shape(gprime), np.shape(prevcontrib)))
    #     if not output:
    #         self.delta = gprime * prevcontrib
    #         return self.delta
    #     else:
    #         self.delta = previousDelta
    #         return gprime*prevcontrib

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
# and there are 1024 inputs, we'd need about 128^1024 / 1024 =~
# 5.93e2154 hidden units.
# Normalize the inputs--try taking their logs or dividing them by 128
# or something. 

#Note: this works:
# a = [1, 2, 3]
# b = [5, 6, 7]
# c = [9, 10, 11]
# map(lambda x, y, z: x+y+z, a, b, c)
