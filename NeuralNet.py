# NeuralNet.py
# A Python implementation of a neural network to distinguish males
# from females. Hoping to get better than that lousy 50% accuracy
# that the Java version had. Using Python cause it's easy and nice.
import argparse, numpy as np, math

class Layer:
    def __init__(self, numNodes):
        self.weights = np.random.uniform(-0.05, 0.05, numNodes)

    def calc_outputs(self, inputs):
        self.output = np.array(map(lambda x : 1/(1 + math.exp(-x)),
                                        np.dot(self.weights, inputs)))
        return self.output  # Badly thought out, having it both ways design.

class Network:
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=
                                   "Takes train, test, and verify options.")
    args.add_argument("-train", nargs=2)
    args.add_argument("-test", action='store')
    args.add_argument("-verify", action='store_true')
    argnames = args.parse_args()
    print(argnames)
