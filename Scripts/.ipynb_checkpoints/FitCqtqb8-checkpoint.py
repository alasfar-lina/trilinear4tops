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
########
stream = open(filename, 'r')#
data = yaml.safe_load(stream)
operator='Cqtqb8'
runs=300000
################################
##############################################################
ll = lambda c4q,ch : mylikelihoodAV(0,0,0,c4q,0,ch, data=data,collider='Run-II',mode='rge',l3mode='linear')
limits1 =[-5., 5., -50, 30]
config = [runs, 0.8, 50]
trace_dir=dirc+'/results/fits/Cqtqb8-Cphi_LHC_RunII_linearl3_rge.pickle'
model= runMCMC((ll), limits1, config=config,trace_dir=trace_dir)
print('done')
##############################################################
ll = lambda c4q,ch : mylikelihoodAV(0,0,0,c4q,0,ch, data=data,collider='Run-II',mode='rge',l3mode='quadratic')
limits1 =[-5., 5., -40, 30]
config = [runs, 0.8, 50]
trace_dir=dirc+'/results/fits/Cqtqb8-Cphi_LHC_RunII_quadl3_rge.pickle'
model= runMCMC((ll), limits1, config=config,trace_dir=trace_dir)
print('done')
##############################################################
print('done All')
ll = lambda c4q,ch :mylikelihoodAV(0,0,0,c4q,0,ch, data=data,collider='Run-II',mode='fin',l3mode='linear')
limits = [-30.0, +30.0, -50, 30]
config = [runs, 0.8, 50]
trace_dir=dirc+'/results/fits/Cqtqb8-Cphi_LHC_RunII_linearl3_fin.pickle'
model= runMCMC((ll), limits, config=config,trace_dir=trace_dir)
print('done')
##############################################################
ll1 = lambda c4q,ch : mylikelihoodAV(0,0,0,c4q,0,ch, data=data,collider='Run-II',mode='fin',l3mode='quadratic')
limits1 = [-30.0, +30.0, -50, 30]
config1 = [runs, 0.8, 50]
trace_dir1=dirc+'/results/fits/Cqtqb8-Cphi_LHC_RunII_quadl3_fin.pickle'
model= runMCMC((ll1), limits1, config=config1,trace_dir=trace_dir1)
##############################################################
print('done All')