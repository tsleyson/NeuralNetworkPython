# NeuralNetTests.py
# Unit tests for the neural network class. Mostly I wanted to play with
# the unittest module, but if we don't need this, we don't need it.
# Note: Ctrl-c-> and Ctrl-c-< to indent in Emacs.
import unittest, NeuralNet, numpy, random

class TestNetwork(unittest.TestCase):
        # Intention in these tests is to learn xor function.
        # As it nears completion, going to add tests for
        # successful learning of xor within some reasonable
        # margin of error. (Just use Network.judgment_on with
        # a suitable decision function to decide the margin
        # of error.)
	def setUp(self):
                self.network = NeuralNet.Network(2, [2, 4, 1], 0.5)
                self.trainingData = [(numpy.array(x[0]), x[1]) 
                                for x in [([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0)]]
        def test_feed(self):
                print(self.network.feed_network(numpy.array([0, 0])))
                for datum in self.trainingData:
                        print(self.network.feed_network(datum[0]))
        def test_propagate(self):
                for datum in self.trainingData:
                        print(datum[0], datum[1])
                        self.network.propagate_back(datum[0], datum[1])
                for datum in self.trainingData:
                        print(self.network.feed_network(datum[0]))
        def test_training(self):
                for i in range(10000):
                        for datum in self.trainingData:
                               self.network.propagate_back(datum[0], datum[1])
                        random.shuffle(self.trainingData)
                for datum in self.trainingData:
                        print(self.network.feed_network(datum[0]), datum[1])
# end TestNetwork

#class TestLayer(unittest.TestCase):
#        def setUp(self):
#                self.layer = NeuralNet.Layer(
                
if __name__ == "__main__":
	unittest.main()
