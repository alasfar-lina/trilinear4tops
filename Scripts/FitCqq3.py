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
operator='Cqq3'
runs=200000
################################
##############################################################
ll = lambda c4q,ch : mylikelihoodAV(0,0,0,0,c4q,ch,data,collider='Run-II',mode='rge',l3mode='linear')
limits1 =[-200., 200., -40, 40]
config = [runs, 0.8, 50]
trace_dir=dirc+'/results/fits/Cqq3-Cphi_LHC_RunII_linearl3_rge.pickle'
model= runMCMC((ll), limits1, config=config,trace_dir=trace_dir)
print('done')
##############################################################
ll = lambda c4q,ch : mylikelihoodAV(0,0,0,0,c4q,ch,data,collider='Run-II',mode='rge',l3mode='quadratic')
limits1 =[-200., 200., -30, 30]
config = [runs, 0.8, 50]
trace_dir=dirc+'/results/fits/Cqq3-Cphi_LHC_RunII_quadl3_rge.pickle'
model= runMCMC((ll), limits1, config=config,trace_dir=trace_dir)
#print('done')
##############################################################
#print('done All')
#ll = lambda c4q,ch :mylikelihoodAV(0,0,0,0,c4q,ch,data,collider='Run-II',mode='fin',l3mode='linear')
#limits = [-35.0, +35.0, -50, 50]
#config = [runs, 0.8, 50]
#trace_dir=dirc+'/results/fits/Cqq3-Cphi_LHC_RunII_linearl3_fin.pickle'
#model= runMCMC((ll), limits, config=config,trace_dir=trace_dir)
#print('done')
##############################################################
#ll1 = lambda c4q,ch : mylikelihoodAV(0,0,0,0,c4q,ch,data,collider='Run-II',mode='fin',l3mode='quadratic')
#limits1 = [-30.0, +30.0, -50, 50]
#config1 = [runs, 0.8, 50]
#trace_dir1=dirc+'/results/fits/Cqq3-Cphi_LHC_RunII_quadl3_fin.pickle'
#model= runMCMC((ll1), limits1, config=config1,trace_dir=trace_dir1)
##############################################################
#print('done All')