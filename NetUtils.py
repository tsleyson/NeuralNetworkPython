# NetUtils.py
# Utilities for the NeuralNet class. Includes:
# datapoint
#  A named tuple for holding correct and calculated values.
# simple_margin
#  Returns a simple success function that calculates whether
#  the error of an output is within the provided margin.
# success_rate
#  A function that takes an array of training data and outputs and
#  calculates the success rate by checking the number of examples
#  that fall within a specified margin of error.
import collections
import numpy as np
import matplotlib.pyplot as plt
import pdb
datapoint = collections.namedtuple('datapoint', 'correct, calculated')

def simple_margin(error=0):
    """
    Given a margin of error, returns a function which returns
    1 if the difference between the correct answer and output
    answer of the input tuple x is within that margin.
    Assumes the input is a datapoint named tuple.
    """
    return lambda x : 1 if -error <= x.correct - x.calculated <= error else 0

def success_rate(data, successfn):
    """
    - trainingdata is a list of tuples or datapoints (correctans, judgment), where
      correctans is an array of correct outputs and judgment is an
      array of what the network output.
    - successfn is a function which takes one of these tuples and 
      returns 1 if we can count the answer as correct, otherwise
      returning 0.
    """
    hits = [successfn(d)for d in data]
    return np.mean(hits)

def errors(data):
    """
    Returns a list of the error in each datum.
    """
    ret = [x.correct - x.calculated for x in data]
    return ret

def compare_plot(data, calculated):
    """
    Plots a set of data meant to approximate a continuous
    function alongside that function.
    data includes the input and intended output.
    calculated gives the calculated output.
    Only works for functions in two-dimensional space.
    """
    inputvals = np.array([d[0] for d in data])
    desired = np.array([d[1] for d in data])
    #print(inputvals)
    #print(desired)
    plt.plot(inputvals, desired, color="green", label="desired")
    plt.plot(inputvals, calculated, color="blue", label="generated data")
    plt.legend(loc="upper left")
    plt.show()
    
