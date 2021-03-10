# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:08:59 2016

@author: bitzer
"""

import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import warn


#%% define pdf of Gauss-prob distribution
def gaussprobpdf(y, mu, sigma2, width=1.0, shift=0.0):
    """pdf of the distribution resulting from transforming a Gaussian random
       variable through the standard normal cumulative density function.
       
       mu, sigma2 are the mean and variance of the Gaussian random variable x
       the transformed random variable y is:
       
       y = norm.cdf(x) * width + shift
    """
    if sigma2 == 1.0:
        sigma2 -= 1e-15
        warn('subtracted 1e-15 from sigma2=1.0 to avoid division by 0')
        
    return np.exp((sigma2 - 1) / 2 / sigma2 * (
        scipy.stats.norm.ppf((y - shift) / width) - mu / (1 - sigma2)) ** 2 + 
        mu**2 / (1 - sigma2) / 2 ) / np.sqrt(sigma2) / width
        
        
#%% some tests
def check_gaussprobpdf(mu=0.0, sigma=1.0):
    g_samples = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=10000)
    p_samples = scipy.stats.norm.cdf(g_samples)
    
    plt.figure()
    ax = sns.distplot(p_samples)
    lower, upper = ax.get_xlim()
    yy = np.linspace(lower, upper, 1000)
    ax.plot(yy, gaussprobpdf(yy, mu, sigma**2))