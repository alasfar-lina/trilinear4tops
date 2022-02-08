import numpy as np
import yaml
import pymc3 as pm
import math
import arviz as az
import theano
import theano.tensor as tensor
import pandas as pd
from scipy import optimize
import pickle 
######
dirc= '/beegfs/desy/user/lalasfar/trilinear4tops'
import sys
sys.path.append(dirc+'/HelpherFunctions/')
from Chi2Allvar import *
from mcmc import *
######
filename =dirc+"/results/data.yaml"
########
stream = open(filename, 'r')#
data = yaml.safe_load(stream)
operator='Cqtqb8'
################################

ll = lambda c4q,ch :mylikelihoodAV(0,0,0,c4q,0,ch,data,collider='HL-LHC',mode='rge',l3mode='linear')
limits = [-3.5, +3.5, -30, 30]
config = [100000, 0.8, 50]
trace_dir=dirc+'/results/fits/Cqtqb8_HL-LHC_linearl3_rge.pickle'
model= runMCMC((ll), limits, config=config,trace_dir=trace_dir)
#
print('done')

ll1 = lambda c4q,ch : mylikelihoodAV(0,0,0,c4q,0,ch,data,collider='HL-LHC',mode='rge',l3mode='quadratic')
limits = [-3.5, +3.5, -30, 30]
config = [100000, 0.8, 50]
trace_dir=dirc+'/results/fits/Cqtqb8_HL-LHC_quadl3_rge.pickle'
model= runMCMC((ll1), limits, config=config,trace_dir=trace_dir)
print('done')
##############################################################

print('done All')