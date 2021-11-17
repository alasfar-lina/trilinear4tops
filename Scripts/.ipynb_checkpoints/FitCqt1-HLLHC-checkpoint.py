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
from Chi2 import *
from mcmc import *
######
filename =dirc+"/results/data.yaml"
########
stream = open(filename, 'r')#
data = yaml.safe_load(stream)
operator='Cqt1'
################################
ll = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['HL-LHC'], HiggsChannels=['ggf','vbf','wh','zh','ttxhhllhc'],TopChannels=None,mode='rge',l3mode='linear',linearmu=True) 
limits = [-2.5, +2.5, -35, 35]
config = [200000, 0.8, 50]
trace_dir=dirc+'/results/fits/Cq1_HL-LHC_linearl3_rge.pickle'
model= runMCMC((ll), limits, config=config,trace_dir=trace_dir)


ll = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['HL-LHC'], HiggsChannels=['ggf','vbf','wh','zh','ttxhhllhc'],TopChannels=None,mode='rge',l3mode='quadratic',linearmu=True) 
limits = [-2.5, +2.5, -35, 35]
config = [200000, 0.8, 50]
trace_dir=dirc+'/results/fits/Cq1_HL-LHC_quadl3_rge.pickle'
model= runMCMC((ll), limits, config=config,trace_dir=trace_dir)
print('done')
##############################################################


print('done All')