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
operator='Cqt8'
################################


ll = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['ATLAS','CMS'],HiggsChannels=['ggf','vbf','ttxh','vh','wh','zh'],TopChannels=None,mode='rge',l3mode='linear',linearmu=True)
limits = [-35.5, 5.5, -60, 30]
config = [200000, 0.8, 50]
trace_dir=dirc+'/results/fits/Cq8_LHC_RunII_linearl3_rge.pickle'
model= runMCMC((ll), limits, config=config,trace_dir=trace_dir)
print('done')
##############################################################
ll1 = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['ATLAS','CMS'],HiggsChannels=['ggf','vbf','ttxh','vh','wh','zh'],TopChannels=None,mode='rge',l3mode='quadratic',linearmu=True)
limits1 = [-15.5, 10.5, -25, 10]
config1 = [200000, 0.8, 50]
trace_dir1=dirc+'/results/fits/Cq8_LHC_RunII_quadl3_rge.pickle'
model= runMCMC((ll1), limits1, config=config1,trace_dir=trace_dir1)
print('done All')