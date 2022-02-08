import numpy as np
import matplotlib as mtlp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import yaml
import pymc3 as pm
import arviz as az
import theano
import pandas as pd
from matplotlib import rc
from scipy import optimize
import seaborn as sns
import pickle 
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as mpl_patches
dirc= '/beegfs/desy/user/lalasfar/trilinear4tops'
######
import sys
sys.path.append(dirc+'/HelpherFunctions/')
from Chi2Allvar import *
######
filename =dirc+"/results/data.yaml"
########
stream = open(filename, 'r')#
data = yaml.safe_load(stream)
#
colpastil = ['#9cadce','#937eba','#f09494','#72bbd0','#52b2cf','#ffafcc','#d3ab9e' ]
NBINS = 100
LambdaNP2 = 1e+3**2
v4 = 246.**4
v3 = 246.**3
mh2 = 125.1**2
sqrt_2 = np.sqrt(2.0)
NBINS = 50
CF=4/3
Nc=3


def mode(x):
    """ Finds the mode of x
        argument:
            x: an array
    """
    n, bins = np.histogram(x, bins=101)
    m = np.argmax(n)
    m = (bins[m] + bins[m-1])/2.
    return m

def multimode(x, n, hdi_prob):
    """ Finds all the modes in the distribution
        arguments:
            x: the array for the distribution
            n: the identifier for the variable
    """
    md = az.hdi(x, hdi_prob=hdi_prob, multimodal=False)
    if len(md) < 2 and n > 1:
        return np.NaN
    else:
        return md[n%2]
    
def mode(x):
    # Function to find mode of an array x
    n, bins = np.histogram(x, bins=101)
    m = np.argmax(n)
    m = (bins[m] + bins[m-1])/2.
    return m

def minimize(likelihood, guess):
    """ Minimizing routine for finding global mode
    argument:
        likelihood: the likelihood function
        guess: the guess for the mode, [r, theta]
    """
    res = optimize.minimize(lambda x: -likelihood(x[0], x[1]), guess, method='BFGS', tol=1e-6)
    return res
unity = lambda x : x
stats_func_2 = {
        'b0': lambda x: multimode(x, 0, 0.9545),
        'b1': lambda x: multimode(x, 1, 0.9545),
        }
stats_func_1 = {
        'b0': lambda x: multimode(x, 0, 0.6827),
        'b1': lambda x: multimode(x, 1, 0.6827),
        }

def runMCMC4(likelihood, limits, trace_dir='', config=[], fit=True):
    """ pyMC3 MCMC run
        argument:
            likelihood: the likelihood function
            limits: an array of the limits for the parameters [r_lowers, r_upper, theta_lower, theta_upper]
            trace_dir: the directory to which the MCMC traces are saves. '' implies none
            config: the setup for the MCMC. [MCMC smaple size, target_accept, chains]
            fit: bolean for determining whether to run the fit
        returns:
            trace: if fit is true it returns the trace
            model;if fit is false it returns the model
    """
    with pm.Model() as model:
        k1 = pm.Uniform('k1', lower=limits[0], upper=limits[1])
        k2 = pm.Uniform('k2', lower=limits[2], upper=limits[3])
        k3 = pm.Uniform('k3', lower=limits[4], upper=limits[5])
        k4 = pm.Uniform('k4', lower=limits[6], upper=limits[7])

        like = pm.Potential('like', likelihood(k1, k2, k3, k4))
        
    if fit:
        with model:
       #     theano.config.compute_test_value = "off"
            trace = pm.sample(config[0], tune=int(np.max([1000,config[0]/5])), cores=4, target_accept=config[1], chains=config[2], init='advi_map')
#             print(az.summary(trace, round_to=5)) # Turn on to print summary
       #     theano.config.compute_test_value = "off"
           # if trace_dir != '': pm.save_trace(trace=trace, directory=trace_dir, overwrite=True)
            with open(trace_dir, 'wb') as buff:
                pickle.dump({'model': model, 'trace': trace}, buff)
        return trace, model
    return model

def mainquad():
    llCqtm =lambda Cqu1,Cqu8,Cquqbp,CH :mylikelihoodAV(Cqu1,Cqu8,
                                                   1/2/(2*Nc+1)*Cquqbp,+1/CF*Cquqbp/2,0,CH,
                                                    data=data,collider='Run-II',mode='rge',l3mode='quadratic')
    limits = [-8., 12,-25.0,40.,-3.,5.0, -35, 25] #for quad run2
    config = [300000, 0.8, 50]
    trace_dir=dirc+'/results/fits/4paramfit_LHC_RunII_l3Q_rge.pickle'
    trace_1, model_1 = runMCMC4(llCqtm, limits, config=config,trace_dir=trace_dir)

def mainlin():
    llCqtm =lambda Cqu1,Cqu8,Cquqbp,CH :mylikelihoodAV(Cqu1,Cqu8,
                                                   1/2/(2*Nc+1)*Cquqbp,+1/CF*Cquqbp/2,0,CH,
                                                   data=data,collider='Run-II',mode='rge',l3mode='linear')
    #limits = [-8., 12,-25.0,40.,-3.,5.0, -35, 25] #for quad run2
    limits = [-8., 12,-30.0,55.,-5.5,5.5, -90, 50] #for lin run2
    #limits = [-4., 4.,-15.,15.,-4.,4., -30, 30]
    config = [300000, 0.8, 50]
    trace_dir=dirc+'/results/fits/4paramfit_LHC_RunII_l3L_rge.pickle'
    trace_1, model_1 = runMCMC4(llCqtm, limits, config=config,trace_dir=trace_dir)    
    
if __name__ =='__main__':
    mainquad()
    mainlin()