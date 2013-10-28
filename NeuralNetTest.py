# NeuralNetTests.py
# Unit tests for the neural network class. Mostly I wanted to play with
# the unittest module, but if we don't need this, we don't need it.
# Note: Ctrl-c-> and Ctrl-c-< to indent in Emacs.
import unittest
import NeuralNet
import NetUtils
import numpy
import random
import math
from math import sin
from math import radians
from NetUtils import datapoint

def dtanhdx(x):
        """
        Derivative of tanh, designed for use on a whole numpy array.
        """
        return 1 - numpy.array(map(lambda y: math.tanh(y)**2, x))

class TestUtils(unittest.TestCase):
	def setUp(self):
		self.f = NetUtils.simple_margin(error=0.4)
		self.data = [datapoint(x[0], x[1]) for x in [(0.7, 1), (2, 1), 
						    (0.4, 1), (0.4, 0.4), (0.3, 2.2),
						    (42, 0), (0.23, 13), (12.3, 11.92)]]
	
	def test_simple_margin(self):
		tests = [(self.f(datapoint(1.4, 1)), 1),
			 (self.f(datapoint(1.2, 1)), 1),
			 (self.f(datapoint(2, 1)), 0)]
		count = 0
		for test in tests:
			print(test[0])
			count += 1
			assert test[0] == test[1], "Wrong margin bounds on {0}".format(count)
	def test_success_rate(self):
		rate = NetUtils.success_rate(self.data, self.f)
		print(rate)
		assert rate == 0.375, "Success rate is wrong."
	def test_errors(self):
		assert NetUtils.errors(self.data) == [-0.30000000000000004, 1, -0.6, 0.0,
						       -1.9000000000000001, 42, -12.77,
						       0.3800000000000008]
# end TestNetUtils

class TestNetwork:
        def __init__(self):
                self.network = None
                self.trainingData = None
                raise NotImplementedError
        def test_feed(self):
                for datum in self.trainingData:
                        self.network.feed_network(datum[0])
                        # Was printing; now just run and make sure
                        # it doesn't throw an error.
        def test_propagate(self):
#                self.network.print_me()                
                for datum in self.trainingData:
                        #print(datum[0], datum[1])
                        self.network.propagate_back(datum[0], datum[1])
                for datum in self.trainingData:
                        self.network.feed_network(datum[0])
        def test_training(self):
                for i in range(10000):
                        for datum in self.trainingData:
                                self.network.propagate_back(datum[0], datum[1])
                        random.shuffle(self.trainingData)
                print("Results of the training on {0}:".format(self.network))
                for datum in self.trainingData:
                        judgment = self.network.judgment_on(self.network.feed_network(datum[0]),
                                                            lambda x: 0 if x[0] < 0.5 else 1)
                        print("{0} xor {1} = {2}".format(datum[0][0], datum[0][1], judgment))
# end TestNetwork

class TestNetwork331(unittest.TestCase, TestNetwork):
        # Intention in these tests is to learn xor function.
        # As it nears completion, going to add tests for
        # successful learning of xor within some reasonable
        # margin of error. (Just use Network.judgment_on with
        # a suitable decision function to decide the margin
        # of error.)
        # 10/17/2013: The 3-3-1 network successfully learns the xor function after
        # 10,000 back propagations with learning rate 0.5 and the
        # weights set to random values in the range -0.5, 0.5. I had
        # a feeling those values were too small and were stopping the
        # network from converging in reasonable time. Yay!
        # (It also learns it after 1000. Probably by overfitting, but
        # that's something, right?)
        # (And also at 500, but not at 100.)

	def setUp(self):
                # Testing two different configurations for an xor network.
                self.network = NeuralNet.Network([3, 3, 1], learningrate=0.5, initInterval=0.5)
                self.trainingData = [(numpy.array(x[0]), numpy.array(x[1]))
                                     for x in [([0,0,1],0), ([0,1,1],1), ([1,0,1],1), ([1,1,1],0)]]

class TestNetwork241(TestNetwork331):
        """
        This network also learns the xor function using a 2-4-1 network without
        a bias.
        """
        # Currently the dimensions are wrong in calc_delta for the
        # 2-4-1 network, so it doesn't work.
        # The two-element error (prevcontrib) vector seems right; we
        # should have one delta for each of the two neurons previous
        # layer (setting aside that it's the input layer). 
        def setUp(self):
                self.network = NeuralNet.Network([2, 4, 1], learningrate=0.5)
                self.trainingData = [(numpy.array(x[0]), numpy.array(x[1]))
                               for x in [([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0)]]
        # end setUp
# end TestNetwork241

class TestNetwork341(unittest.TestCase, TestNetwork):
        """
        This network learns xor with a bias and four hidden nodes.
        """
        def setUp(self):
                self.network = NeuralNet.Network([3, 4, 1], learningrate=0.5, initInterval=0.3)
                self.trainingData = [(numpy.array(x[0]), numpy.array(x[1]))
                                for x in [([0,0,1],0), ([0,1,1],1), ([1,0,1],1), ([1,1,1],0)]]
# end TestNetwork341

# class TestNetworkSin(unittest.TestCase, TestNetwork):
#         """
#         This network learns sin x, for x in degrees. (In radians if that fails.)
#         """
#         def setUp(self):
#                 self.network = NeuralNet.Network([1, 13, 1], learningrate=0.1, initInterval=0.05,
#                                                  activation=math.tanh,
#                                                  derivative=dtanhdx)
#                 self.trainingData = [(numpy.array([radians(x)]), numpy.array([sin(radians(x))]))
#                                      for x in range(0, 361)]
#         def test_training(self):
#                 for i in range(1000):
#                         for datum in self.trainingData:
#                                 self.network.propagate_back(datum[0], datum[1])
#                         random.shuffle(self.trainingData)
#                 print("Results of the training on {0}:".format(self.network))
#                 checklist = []
#                 for datum in self.trainingData:
#                         judgment = self.network.feed_network(datum[0])
#                         checklist.append(NetUtils.datapoint(correct=datum[0][0], 
# 							    calculated=judgment))
#                 print("Success rate: {0:<20}\nAverage Error: {0:<20}".format(
# 				NetUtils.success_rate(checklist,NetUtils.simple_margin(0.2)),
# 				numpy.mean(NetUtils.errors(checklist))))
# 		NetUtils.compare_plot(map(lambda t: (t[0][0], t[1][0]), self.trainingData), 
# 					  [c.calculated for c in checklist])

#matplotlib.pyplot.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
#[r'$-\pi$', r'$-\pi/2$, r'$0$', r'$+\pi/2$', r'$+\pi$'])

# end TestNetwork 121

if __name__ == "__main__":
	unittest.main()
