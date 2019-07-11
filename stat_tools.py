import random
import numpy as np
 
def random_sample(probability, n=100):
    " Returns a set of n booleans based on the decimal probability of getting a True value "
    return np.array([random.random() < probability for i in range(n)]) 
