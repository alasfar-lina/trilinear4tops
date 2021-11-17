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
CF= 4/3
Nc=3
################################


llCqtp =lambda Cqup,CH :mylikelihoodAV(0.0,0.0,1/2/(2*Nc+1)*Cqup,+1/CF*Cqup/2,CH,data,experiments=['CMS','ATLAS'],
                                                                           HiggsChannels['ggf','vbf','ttxh','vh','zh','wh']
                                                                           ,TopChannels=None,
                                                                           l3mode='linear',linearmu=True)
limits = [-5., 5., -30, 30]
config = [150000, 0.8, 50]
trace_dir='../results/fits/CQtQbp_linearl3_linearmu.pickle'
model= runMCMC(llCqtp, limits, config=config,trace_dir=trace_dir)
print('done')
##############################################################
llCqtp =lambda Cqup,CH :mylikelihoodAV(0.0,0.0,1/2/(2*Nc+1)*Cqup,+1/CF*Cqup/2,CH,data,experiments=['CMS','ATLAS'],
                                                                           HiggsChannels['ggf','vbf','ttxh','vh','zh','wh']
                                                                           ,TopChannels=None,
                                                                           l3mode='linear',linearmu=False)
limits = [-5., 5., -30, 30]
config = [150000, 0.8, 50]
trace_dir='../results/fits/CQtQbp_linearl3.pickle'
model= runMCMC(llCqtp, limits, config=config,trace_dir=trace_dir)
print('done')
##############################################################