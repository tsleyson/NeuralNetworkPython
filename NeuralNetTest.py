# NeuralNetTests.py
# Unit tests for the neural network class. Mostly I wanted to play with
# the unittest module, but if we don't need this, we don't need it.
import unittest, NeuralNet, numpy

class TestNetwork(unittest.TestCase):
	def setUp(self):
                self.network = NeuralNet.Network(3, [2, 2, 2, 2])
        def test_feed(self):
                print(self.network.feed_network(numpy.array([0, 0])))
if __name__ == "__main__":
	unittest.main()
