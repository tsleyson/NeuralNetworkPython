# NeuralNetTests.py
# Unit tests for the neural network class. Mostly I wanted to play with
# the unittest module, but if we don't need this, we don't need it.
import unittest, NeuralNet

class NetTestBasic(unittest.TestCase):
	def setUp(self):
		self.network = NeuralNet.Network(4, [2, 3, 1])

class TestFeedByPrinting(NetTestBasic):
	def runTest(self):
		print(self.network.feed_network(np.array[0,1]))

if __name__ == "__main__":
	n = TestFeedByPrinting()
	n.runTest()
