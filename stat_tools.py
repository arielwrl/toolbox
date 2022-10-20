import random
import numpy as np
from sklearn.cluster import MeanShift

 
def random_sample(probability, n=100):
    " Returns a set of n booleans based on the decimal probability of getting a True value "
    return np.array([random.random() < probability for i in range(n)]) 


def gini(x):

    mad = np.abs(np.subtract.outer(x, x)).mean()

    rmad = mad/np.mean(x)

    g = 0.5 * rmad
    return g


def int_to_bool_list(num):

   bin_string = format(num, '05b')

   return [x == '1' for x in bin_string[::-1]]


def find_maximum(pdf):
    
    """
    
    Finds maximum value of a pdf
    
    """
    
    nanflag = ~np.isnan(pdf)
    
    bins, edges = np.histogram(pdf[nanflag], bins=200)
    
    max_index = np.argmax(bins)
    
    max_value = (edges[max_index] + edges[max_index+1])/2
    
    return max_value


def find_pdf_peaks(pdf, bandwidth=0.05):
    
    pdf = pdf[~np.isnan(pdf)]
    
    outlier_filter = (pdf > np.percentile(pdf, 5)) & (pdf < np.percentile(pdf, 95)) 
    
    filtered_pdf = pdf[outlier_filter]
    
    clustering = MeanShift(bandwidth=bandwidth).fit(filtered_pdf.reshape(-1,1))
    
    return clustering.cluster_centers_.ravel()