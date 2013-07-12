# NeuralNet.py
# A Python implementation of a neural network to distinguish males
# from females. Hoping to get better than that lousy 50% accuracy
# that the Java version had. Using Python cause it's easy and nice.
import sys, os, argparse, numpy as np

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Takes train, test, and verify options.")
    args.add_argument("-train", nargs=2)
    args.add_argument("-test", action=store)
    args.add_argument("-verify", action=store_true)
    argnames = args.parse_args()
    print(argnames)
