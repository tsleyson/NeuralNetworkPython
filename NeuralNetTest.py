# NeuralNetTests.py
# Unit tests for the neural network class. Mostly I wanted to play with
# the unittest module, but if we don't need this, we don't need it.
# Note: Ctrl-c-> and Ctrl-c-< to indent in Emacs.
import unittest, NeuralNet, numpy, random

class TestNetwork:
        def __init__(self):
                self.network = None
                self.trainingData = None
                raise NotImplementedError
        def test_feed(self):
                for datum in self.trainingData:
                        print(self.network.feed_network(datum[0]))
        def test_propagate(self):
#                self.network.print_me()                
                for datum in self.trainingData:
                        #print(datum[0], datum[1])
                        self.network.propagate_back(datum[0], datum[1])
                for datum in self.trainingData:
                        print(self.network.feed_network(datum[0]))
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

if __name__ == "__main__":
	unittest.main()
