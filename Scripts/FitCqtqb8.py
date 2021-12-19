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
########
stream = open(filename, 'r')#
data = yaml.safe_load(stream)
operator='Cqtqb8'
runs=1000000
################################
##############################################################
#ll = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['ATLAS','CMS'],HiggsChannels=['ggf','vbf','ttxh','vh','wh','zh'],TopChannels=None,mode='rge',l3mode='linear',linearmu=True)
#limits1 =[-10.5, 10.5, -40, 20]
#config = [runs, 0.8, 50]
#trace_dir=dirc+'/results/fits/Cqtqb8_LHC_RunII_linearl3_rge.pickle'
#model= runMCMC((ll), limits1, config=config,trace_dir=trace_dir)
#print('done')
##############################################################
#ll = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['ATLAS','CMS'],HiggsChannels=['ggf','vbf','ttxh','vh','wh','zh'],TopChannels=None,mode='rge',l3mode='quadratic',linearmu=True)
#limits1 =[-10.5, 10.5, -40, 20]
#config = [runs, 0.8, 50]
#trace_dir=dirc+'/results/fits/Cqtqb8_LHC_RunII_quadl3_rge.pickle'
#model= runMCMC((ll), limits1, config=config,trace_dir=trace_dir)
#print('done')
##############################################################
ll1 = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['ATLAS','CMS'],HiggsChannels=['ggf','vbf','ttxh','vh','wh','zh'],TopChannels=None,mode='fin',l3mode='quadratic',linearmu=True)
limits1 =[-25.5, 10.5, -40, 20]
config = [runs, 0.8, 50]
trace_dir1=dirc+'/results/fits/Cqtqb8_LHC_RunII_quadl3_fin.pickle'
model= runMCMC((ll1), limits1, config=config,trace_dir=trace_dir1)
##############################################################
ll1 = lambda c4q,ch : mylikelihood(operator,c4q,ch,data,experiments=['ATLAS','CMS'],HiggsChannels=['ggf','vbf','ttxh','vh','wh','zh'],TopChannels=None,mode='fin',l3mode='linear',linearmu=True)
limits1 =[-25.5, 10.5, -40, 20]
config = [runs, 0.8, 50]
trace_dir1=dirc+'/results/fits/Cqtqb8_LHC_RunII_linearl3_fin.pickle'
model= runMCMC((ll1), limits1, config=config,trace_dir=trace_dir1)
print('done All')
