import numpy as np
import matplotlib as mtlp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from pyik.mplext import plot_hist, uncertainty_ellipse
from scipy.stats import chi2
import yaml
import theano.tensor as tt
import theano
import pymc3 as pm

def rk4(f, x0, y0, x1, n):
    vx = [0] * (n + 1)
    vy = [0] * (n + 1)
    h = (x1 - x0) / float(n)
    vx[0] = x = x0
    vy[0] = y = y0
    for i in range(1, n + 1):
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)
        vx[i] = x = x0 + i * h
        vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vx, vy


def mylikelihood(operator,Cqu1,CH,data,experiments=['ATLAS'],HiggsChannels=['ggf','vbf','ttxh','vh'],TopChannels='smeft_fit_quad',mode='fin',l3mode='linear',linearmu=True):
    
    """ Chi2 function in vectorissed manner
            mu0: vector of the expected poi
            mu1: vector of the observed poi
            err: uncertainties, a vector
            corr: correlation matirx
            ndf: number of defgrees of freedom
    """
 # constants
    lambda2= 1000**2
    v2=1/(2**(1/2)*data['SM']['GF'])
    v= np.sqrt(v2)
    mh=data['SM']['mh']
    pi= np.pi
    ZH= (-9*data['SM']['GF']*data['SM']['mh']**2*(-1 + (2*pi)/(3.*(3.)**.5)))/(16.*(2.)**.5*pi**2)
    xs_zh= data['SM']['xs_zh_14']if experiments ==['HL-LHC'] else  data['SM']['xs_zh_13']
    xs_wh= data['SM']['xs_wh_14']if experiments ==['HL-LHC'] else  data['SM']['xs_wh_13']

    d_gamma_tot_ch = data['kl']['tot_gamma']
    #difference in cross sections from 4 fermion operators
    gamma_gaga_cqu1= data[operator]['gagaos_'+mode]* Cqu1
    gamma_gg_cqu1=data[operator]['htogg_'+mode]* Cqu1
    gamma_bb_cqu1=data[operator]['Hbb_'+mode]* Cqu1
    sigma_gg_cqu1 =data[operator]['ggFos_'+mode]* Cqu1
    sigma_ttH_cqu1= data[operator]['ttH14_'+mode]* Cqu1 if experiments== ['HL-LHC'] else data[operator]['ttH_'+mode]* Cqu1
############# Cphi modifications #####################
######################################################  
    #theoretical input
    C1gaga=data['kl']['gaga']
    C1gg = data['kl']['ggF']
    C1zz =data['kl']['zz']
    C1ww=data['kl']['ww']
    C1ff=data['kl']['ff']
    C1VBF=data['kl']['VBF']
    C1ZH=data['kl']['ZH14']  if experiments== ['HL-LHC'] else data['kl']['ZH'] 
    C1WH=data['kl']['WH']
    C1ttH= data['kl']['ttH14'] if experiments== ['HL-LHC'] else data['kl']['ttH'] 
    C14l=data['kl']['hto4l']
    d_gamma_tot_ch = data['kl']['tot_gamma']
    ############
    cH=CH
    lineafunc = lambda c1: -2*v**4/(mh**2*lambda2)*cH*(c1-2*ZH/(ZH-1))
    quadfunc = lambda c1:lineafunc(c1)+(4.*cH**2*v**8*ZH)/(mh**4*(-1. + ZH)**2*lambda2**2)\
    +(12.*cH**2*v**8*ZH**2)/(mh**4*(-1. + ZH)**2*lambda2**2)

    if l3mode=='linear':
        sigma_ch_gg= lineafunc(C1gg)
        sigma_ch_vbf= lineafunc(C1VBF)
        sigma_ch_zh= lineafunc(C1ZH)
        sigma_ch_wh=lineafunc(C1WH)
        sigma_ch_tth= lineafunc(C1ttH)
        ###
        RGamgaga =lineafunc(C1gaga)+gamma_gaga_cqu1
        RGamgg =lineafunc(C1gg)+gamma_gg_cqu1
        RGamww =lineafunc(C1ww)
        RGamzz =lineafunc(C1zz)
        RGambb =lineafunc(C1ff)+gamma_bb_cqu1
        RGamtata =lineafunc(C1ff)
        RGamff =lineafunc(C1ff)
        mu4leptons= lineafunc(C14l)
    elif l3mode=='quadratic':
        sigma_ch_gg= quadfunc(C1gg)
        sigma_ch_vbf= quadfunc(C1VBF)
        sigma_ch_zh= quadfunc(C1ZH)
        sigma_ch_wh=quadfunc(C1WH)
        sigma_ch_tth= quadfunc(C1ttH)
        ###
        RGamgaga =quadfunc(C1gaga)+gamma_gaga_cqu1
        RGamgg =quadfunc(C1gg)+gamma_gg_cqu1
        RGamww =quadfunc(C1ww)
        RGamzz =quadfunc(C1zz)
        RGambb =quadfunc(C1ff)+gamma_bb_cqu1
        RGamtata =quadfunc(C1ff)
        RGamff =quadfunc(C1ff)
        mu4leptons= quadfunc(C14l) # not used 


############## total Higgs width #####################
######################################################  

    GammaSM_WW= data['Higgs']['BRWWSM']*data['Higgs']['width']
    GammaSM_ZZ= data['Higgs']['BRZZSM']*data['Higgs']['width']
    RGamvv = (RGamzz*GammaSM_ZZ+GammaSM_WW*RGamww)/(GammaSM_WW+GammaSM_ZZ)
    GamHRat = (RGambb*\
    data['Higgs']['BRbbSM'] +RGamff*\
    data['Higgs']['BRccSM'] + RGamtata*\
    data['Higgs']['BRTauTauSM'] + RGamff*\
    data['Higgs']['BRMuMuSM'] + RGamww*\
    data['Higgs']['BRWWSM'] + RGamzz*\
    data['Higgs']['BRZZSM'] + RGamgaga*\
    data['Higgs']['BRgagaSM'] + RGamgg*data['Higgs']['BRggSM'])/(data['Higgs']['BRbbSM'] +
    data['Higgs']['BRccSM'] + data['Higgs']['BRTauTauSM'] + data['Higgs']['BRMuMuSM'] + data['Higgs']['BRWWSM'] + data['Higgs']['BRZZSM'] +
    data['Higgs']['BRgagaSM'] + data['Higgs']['BRggSM'])

    GammaHSM= (data['Higgs']['BRbbSM'] +
    data['Higgs']['BRccSM'] + data['Higgs']['BRTauTauSM'] + data['Higgs']['BRMuMuSM'] + data['Higgs']['BRWWSM'] + data['Higgs']['BRZZSM'] +
    data['Higgs']['BRgagaSM'] + data['Higgs']['BRggSM'])

###########################################################################
# linearisation of the signal strength
###########################################################################
    if linearmu==True:
            mugaga= [(1+sigma_gg_cqu1+sigma_ch_gg) +(RGamgaga)-GamHRat]

            muzz =  [(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamzz)-GamHRat]

            muww =[(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamww)-GamHRat]

            mutata =[(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamtata)-GamHRat]

            mumm =[(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamtata)-GamHRat]

            mubb = [(1+sigma_ch_gg+sigma_gg_cqu1)+(RGambb)-GamHRat]

             # vbf
            vbfmugaga= [(1+sigma_ch_vbf) +(RGamgaga)-GamHRat]
            vbfmuzz =[(1+sigma_ch_vbf)+(RGamzz)-GamHRat]
            vbfmuww =[(1+sigma_ch_vbf)+(RGamww)-GamHRat]
            vbfmutata =[(1+sigma_ch_vbf)+(RGamtata)-GamHRat]
            vbfmumm =[(1+sigma_ch_vbf)+(RGamtata)-GamHRat]
            vbfmubb =[(1+sigma_ch_vbf)+(RGambb)-GamHRat]
             # VH
            sigma_vh = ((sigma_ch_zh*xs_zh)+(sigma_ch_wh*xs_wh))/(xs_wh+xs_zh)
            vhmugaga=[(1+sigma_vh)+(RGamgaga)-GamHRat]
            vhmuzz =[(1+sigma_vh)+(RGamzz)-GamHRat]
            vhmuww =[(1+sigma_vh)+(RGamww)-GamHRat]
            vhmutata =[(1+sigma_vh)+(RGamtata)-GamHRat]
            vhmubb =[(1+sigma_vh)+(RGambb)-GamHRat]
            # ZH
            zhmugaga= [(1+sigma_ch_zh)+(RGamgaga)-GamHRat]
            zhmuzz =[(1+sigma_ch_zh)+(RGamzz)-GamHRat]
            zhmuww =[(1+sigma_ch_zh)+(RGamww)-GamHRat]
            zhmutata =[(1+sigma_ch_zh)+(RGamtata)-GamHRat]
            zhmubb = [(1+sigma_ch_zh)+(RGambb)-GamHRat]
             # WH
            whmugaga= [(1+sigma_ch_wh)+(RGamgaga)-GamHRat]
            whmuzz =[(1+sigma_ch_wh)+(RGamzz)-GamHRat]
            whmuww =[(1+sigma_ch_wh)+(RGamww)-GamHRat]
            whmutata =[(1+sigma_ch_wh)+(RGamtata)-GamHRat]
            whmubb =[(1+sigma_ch_wh)+(RGambb)-GamHRat]
             #ttH
            tthmugaga= [(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamgaga)-GamHRat]
            tthmuzz =[(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamzz)-GamHRat]
            tthmuww =[(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamww)-GamHRat ]         
            tthmuvv =[(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamvv)-GamHRat]
            tthmutata = [(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamtata)-GamHRat]
            tthmubb =[(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGambb)-GamHRat]


###########################################################################
# 
#  Matching to experimental input
#
###########################################################################    

    muExp= np.zeros(32)
    errExp= np.zeros(32)
    num_of_obs=muExp.shape[0]
    corr = np.identity(num_of_obs, dtype = float)

    
###############################################################
#           Run-II LHC
 ###############################################################   
    
    ### Run-II ATLAS #####
    if 'ATLAS' in experiments:
        #gluon fusion
        if 'ggf' in HiggsChannels:
            # gamma gamma
            muExp[0]= data['Bounds']['ATLAS']['ggf']['mu_gaga'] 
            errExp[0]=data['Bounds']['ATLAS']['ggf']['err_gaga'] 
            muTheo=np.array(mugaga)
            # ZZ
            muExp[1]= data['Bounds']['ATLAS']['ggf']['mu_zz'] 
            errExp[1]= data['Bounds']['ATLAS']['ggf']['err_zz'] 
            muTheo=np.concatenate((muTheo,muzz))
            #WW
            muExp[2]= data['Bounds']['ATLAS']['ggf']['mu_ww'] 
            errExp[2]= data['Bounds']['ATLAS']['ggf']['err_ww'] 
            muTheo=np.concatenate((muTheo,muww))
            # tau tau
            muExp[3]= data['Bounds']['ATLAS']['ggf']['mu_tata'] 
            errExp[3]= data['Bounds']['ATLAS']['ggf']['err_tata'] 
            muTheo=np.concatenate((muTheo,mutata))
          ###################  
         # VBF
        if 'vbf' in HiggsChannels:
            muExp[4]= data['Bounds']['ATLAS']['vbf']['mu_gaga'] 
            errExp[4]=data['Bounds']['ATLAS']['vbf']['err_gaga'] 
            muTheo=np.concatenate((muTheo,vbfmugaga))
            # ZZ
            muExp[5]= data['Bounds']['ATLAS']['vbf']['mu_zz'] 
            errExp[5]= data['Bounds']['ATLAS']['vbf']['err_zz'] 
            muTheo=np.concatenate((muTheo,vbfmuzz))
            #WW
            muExp[6]= data['Bounds']['ATLAS']['vbf']['mu_ww'] 
            errExp[6]= data['Bounds']['ATLAS']['vbf']['err_ww'] 
            muTheo=np.concatenate((muTheo,vbfmuww))
            # tau tau
            muExp[7]= data['Bounds']['ATLAS']['vbf']['mu_tata'] 
            errExp[7]= data['Bounds']['ATLAS']['vbf']['err_tata'] 
            muTheo=np.concatenate((muTheo,vbfmutata))
            # b b 
            muExp[8]= data['Bounds']['ATLAS']['vbf']['mu_bb'] 
            errExp[8]= data['Bounds']['ATLAS']['vbf']['err_bb'] 
            muTheo=np.concatenate((muTheo,vbfmubb))
          ###################  
         # t t h
        if 'ttxh' in HiggsChannels:
            muExp[9]= data['Bounds']['ATLAS']['ttxh']['mu_gaga'] 
            errExp[9]=data['Bounds']['ATLAS']['ttxh']['err_gaga'] 
            muTheo=np.concatenate((muTheo,tthmugaga))
            # VV
            muExp[10]= data['Bounds']['ATLAS']['ttxh']['mu_vv'] 
            errExp[10]= data['Bounds']['ATLAS']['ttxh']['err_vv'] 
            muTheo=np.concatenate((muTheo,tthmuvv))
            # tau tau
            muExp[11]= data['Bounds']['ATLAS']['ttxh']['mu_tata'] 
            errExp[11]= data['Bounds']['ATLAS']['ttxh']['err_tata'] 
            muTheo=np.concatenate((muTheo,tthmutata))
            #bb
            muExp[12]= data['Bounds']['ATLAS']['ttxh']['mu_bb'] 
            errExp[12]= data['Bounds']['ATLAS']['ttxh']['err_bb'] 
            muTheo=np.concatenate((muTheo,tthmubb))
          ################### 
          # vh
        if 'vh' in HiggsChannels:
            muExp[13]= data['Bounds']['ATLAS']['vh']['mu_gaga'] 
            errExp[13]=data['Bounds']['ATLAS']['vh']['err_gaga']
            muTheo=np.concatenate((muTheo,vhmugaga))
            # ZZ
            muExp[14]= data['Bounds']['ATLAS']['vh']['mu_zz'] 
            errExp[14]= data['Bounds']['ATLAS']['vh']['err_zz'] 
            muTheo=np.concatenate((muTheo,vhmuzz))
            #bb
            muExp[15]= data['Bounds']['ATLAS']['vh']['mu_bb'] 
            errExp[15]= data['Bounds']['ATLAS']['vh']['err_bb'] 
            muTheo=np.concatenate((muTheo,vhmubb))
        ################### 
          #setting up correlations 
        ### gamma gamma between ggf and vbf
        corr[0,4]= data['Bounds']['ATLAS']['corr_ggfvbf_gaga']      
        corr[4,0]=corr[0,4]    
        ### ZZ between ggf and vbf
        corr[1,5]= data['Bounds']['ATLAS']['corr_ggfvbf_zz']   
        corr[5,1]=corr[1,5]   
        ### WW between ggf and vbf
        corr[2,6]=data['Bounds']['ATLAS']['corr_ggfvbf_ww']
        corr[6,2]=corr[2,6]  
        ### tata between ggf and vbf
        corr[3,7]= data['Bounds']['ATLAS']['corr_ggfvbf_tata']
        corr[7,3]=corr[3,7]   
        ### ZZ between ggf and vh
        corr[1,14]=data['Bounds']['ATLAS']['corr_ggf_vh_zz']               
        corr[14,1]=corr[0,13]   
        ### t t h between ta ta and VV
        corr[11,10]=data['Bounds']['ATLAS']['corr_tth_tatazz']               
        corr[10,11]=corr[11,10]     
    
    
    
 ### Run-II CMS #####
    if 'CMS' in experiments:
        #gluon fusion
        if 'ggf' in HiggsChannels:
            # gamma gamma
            muExp[16]= data['Bounds']['CMS']['ggf']['mu_gaga'] 
            errExp[16]=data['Bounds']['CMS']['ggf']['err_gaga'] 
            muTheo=np.concatenate((muTheo,mugaga))
            # ZZ
            muExp[17]= data['Bounds']['CMS']['ggf']['mu_zz'] 
            errExp[17]= data['Bounds']['CMS']['ggf']['err_zz'] 
            muTheo=np.concatenate((muTheo,muzz))
            #WW
            muExp[18]= data['Bounds']['CMS']['ggf']['mu_ww'] 
            errExp[18]= data['Bounds']['CMS']['ggf']['err_ww']
            muTheo=np.concatenate((muTheo,muww))
            # tau tau
            muExp[19]= data['Bounds']['CMS']['ggf']['mu_tata'] 
            errExp[19]= data['Bounds']['CMS']['ggf']['err_tata'] 
            muTheo=np.concatenate((muTheo,mutata))
          ###################  
         # VBF
        if 'vbf' in HiggsChannels:
            muExp[20]= data['Bounds']['CMS']['vbf']['mu_gaga'] 
            errExp[20]=data['Bounds']['CMS']['vbf']['err_gaga'] 
            muTheo=np.concatenate((muTheo,vbfmugaga))

            # ZZ
            muExp[21]= data['Bounds']['CMS']['vbf']['mu_zz'] 
            errExp[21]= data['Bounds']['CMS']['vbf']['err_zz']
            muTheo=np.concatenate((muTheo,vbfmuzz))
            #WW
            muExp[22]= data['Bounds']['CMS']['vbf']['mu_ww'] 
            errExp[22]= data['Bounds']['CMS']['vbf']['err_ww']
            muTheo=np.concatenate((muTheo,vbfmuww))
            # tau tau
            muExp[23]= data['Bounds']['CMS']['vbf']['mu_tata'] 
            errExp[23]= data['Bounds']['CMS']['vbf']['err_tata'] 
            muTheo=np.concatenate((muTheo,vbfmutata))
          ###################  
         # t t h
        if 'ttxh' in HiggsChannels:
            muExp[24]= data['Bounds']['CMS']['ttxh']['mu_gaga'] 
            errExp[24]=data['Bounds']['CMS']['ttxh']['err_gaga']
            muTheo=np.concatenate((muTheo,tthmugaga))
          ################### 
          # vh
        if 'vh' in HiggsChannels:
            muExp[25]= data['Bounds']['CMS']['vh']['mu_gaga'] 
            errExp[25]=data['Bounds']['CMS']['vh']['err_gaga']
            muTheo=np.concatenate((muTheo,vhmugaga))
     
            # ZZ
            muExp[26]= data['Bounds']['CMS']['vh']['mu_zz'] 
            errExp[26]= data['Bounds']['CMS']['vh']['err_zz']
            muTheo=np.concatenate((muTheo,vhmuzz))
            #WW
            muExp[27]= data['Bounds']['CMS']['vh']['mu_ww'] 
            errExp[27]= data['Bounds']['CMS']['vh']['err_ww'] 
            muTheo=np.concatenate((muTheo,vhmuww))
         ################### 
          # zh
        if 'zh' in HiggsChannels:
            #ta ta
            muExp[28]= data['Bounds']['CMS']['zh']['mu_tata'] 
            errExp[28]= data['Bounds']['CMS']['zh']['err_tata'] 
            muTheo=np.concatenate((muTheo,zhmutata))
            # b b
            muExp[29]= data['Bounds']['CMS']['zh']['mu_bb'] 
            errExp[29]= data['Bounds']['CMS']['zh']['err_bb']
            muTheo=np.concatenate((muTheo,zhmubb))
        ################### 
          # wh
        if 'wh' in HiggsChannels:
            #ta ta
            muExp[30]= data['Bounds']['CMS']['wh']['mu_tata'] 
            errExp[30]= data['Bounds']['CMS']['wh']['err_tata'] 
            muTheo=np.concatenate((muTheo,whmutata))
            # b b
            muExp[31]= data['Bounds']['CMS']['wh']['mu_bb'] 
            errExp[31]= data['Bounds']['CMS']['wh']['err_bb']
            muTheo=np.concatenate((muTheo,whmubb))

###############################################################
#            HL-LHC 
 ###############################################################   
 ### HL-LHC CMS #####
    if 'HL-LHC' in experiments:
        #gluon fusion
            # gamma gamma
        muExp[0]= data['Bounds']['HL-LHC']['ggf']['mu_gaga'] 
        errExp[0]=data['Bounds']['HL-LHC']['ggf']['err_gaga'] 
        muTheo=np.array(mugaga)
            # ZZ
        muExp[1]= data['Bounds']['HL-LHC']['ggf']['mu_zz'] 
        errExp[1]= data['Bounds']['HL-LHC']['ggf']['err_zz']
        muTheo=np.concatenate((muTheo,muzz))
            #WW
        muExp[2]= data['Bounds']['HL-LHC']['ggf']['mu_ww'] 
        errExp[2]= data['Bounds']['HL-LHC']['ggf']['err_ww'] 
        muTheo=np.concatenate((muTheo,muww))
            # tau tau
        muExp[3]= data['Bounds']['HL-LHC']['ggf']['mu_tata'] 
        errExp[3]= data['Bounds']['HL-LHC']['ggf']['err_tata']
        muTheo=np.concatenate((muTheo,mutata))
            # b b
        muExp[4]= data['Bounds']['HL-LHC']['ggf']['mu_bb'] 
        errExp[4]= data['Bounds']['HL-LHC']['ggf']['err_bb'] 
        muTheo=np.concatenate((muTheo,mubb))
             # mm
        muExp[5]= data['Bounds']['HL-LHC']['ggf']['mu_mm'] 
        errExp[5]= data['Bounds']['HL-LHC']['ggf']['err_mm'] 
        muTheo=np.concatenate((muTheo,mumm))
          ###################  
         # VBF
        muExp[6]= data['Bounds']['HL-LHC']['vbf']['mu_gaga'] 
        errExp[6]=data['Bounds']['HL-LHC']['vbf']['err_gaga'] 
        muTheo=np.concatenate((muTheo,vbfmugaga))
            # ZZ
        muExp[7]= data['Bounds']['HL-LHC']['vbf']['mu_zz'] 
        errExp[7]= data['Bounds']['HL-LHC']['vbf']['err_zz'] 
        muTheo=np.concatenate((muTheo,vbfmuzz))
            #WW
        muExp[8]= data['Bounds']['HL-LHC']['vbf']['mu_ww'] 
        errExp[8]= data['Bounds']['HL-LHC']['vbf']['err_ww'] 
        muTheo=np.concatenate((muTheo,vbfmuww))
            # tau tau
        muExp[9]= data['Bounds']['HL-LHC']['vbf']['mu_tata'] 
        errExp[9]= data['Bounds']['HL-LHC']['vbf']['err_tata'] 
        muTheo=np.concatenate((muTheo,vbfmutata))
            # mu mu 
        muExp[10]= data['Bounds']['HL-LHC']['vbf']['mu_mm'] 
        errExp[10]= data['Bounds']['HL-LHC']['vbf']['err_mm'] 
        muTheo=np.concatenate((muTheo,vbfmumm))

            
                  ################### 
          # wh
        muExp[11]= data['Bounds']['HL-LHC']['wh']['mu_gaga'] 
        errExp[11]=data['Bounds']['HL-LHC']['wh']['err_gaga'] 
        muTheo=np.concatenate((muTheo,whmugaga))
            # ZZ
        muExp[12]= data['Bounds']['HL-LHC']['wh']['mu_zz'] 
        errExp[12]= data['Bounds']['HL-LHC']['wh']['err_zz'] 
        muTheo=np.concatenate((muTheo,whmuzz))
            #WW
        muExp[13]= data['Bounds']['HL-LHC']['wh']['mu_ww'] 
        errExp[13]= data['Bounds']['HL-LHC']['wh']['err_ww']
        muTheo=np.concatenate((muTheo,whmuww))
             # b b
        muExp[14]= data['Bounds']['HL-LHC']['wh']['mu_bb'] 
        errExp[14]= data['Bounds']['HL-LHC']['wh']['err_bb'] 
        muTheo=np.concatenate((muTheo,whmubb))
                  ################### 
          # zh
        muExp[15]= data['Bounds']['HL-LHC']['zh']['mu_gaga'] 
        errExp[15]=data['Bounds']['HL-LHC']['zh']['err_gaga'] 
        muTheo=np.concatenate((muTheo,whmugaga))
            # ZZ
        muExp[16]= data['Bounds']['HL-LHC']['zh']['mu_zz'] 
        errExp[16]= data['Bounds']['HL-LHC']['zh']['err_zz'] 
        muTheo=np.concatenate((muTheo,zhmuzz))
            #WW
        muExp[17]= data['Bounds']['HL-LHC']['zh']['mu_ww'] 
        errExp[17]= data['Bounds']['HL-LHC']['zh']['err_ww']
        muTheo=np.concatenate((muTheo,zhmuww))
             # b b
        muExp[18]= data['Bounds']['HL-LHC']['zh']['mu_bb'] 
        errExp[18]= data['Bounds']['HL-LHC']['zh']['err_bb']
        muTheo=np.concatenate((muTheo,zhmubb))
           ###################
         # t t h
        #gamma gamma
        muExp[19]= data['Bounds']['HL-LHC']['ttxh']['mu_gaga'] 
        errExp[19]=data['Bounds']['HL-LHC']['ttxh']['err_gaga']
        muTheo=np.concatenate((muTheo,tthmugaga))
       #ZZ
        muExp[20]= data['Bounds']['HL-LHC']['ttxh']['mu_zz'] 
        errExp[20]=data['Bounds']['HL-LHC']['ttxh']['err_zz'] 
        muTheo=np.concatenate((muTheo,tthmuzz))
               #WW
        muExp[21]= data['Bounds']['HL-LHC']['ttxh']['mu_ww'] 
        errExp[21]=data['Bounds']['HL-LHC']['ttxh']['err_ww'] 
        muTheo=np.concatenate((muTheo,tthmuww))

        #b b 
        muExp[22]= data['Bounds']['HL-LHC']['ttxh']['mu_bb'] 
        errExp[22]=data['Bounds']['HL-LHC']['ttxh']['err_bb'] 
        muTheo=np.concatenate((muTheo,tthmubb))
        # ta ta 
        muExp[23]= data['Bounds']['HL-LHC']['ttxh']['mu_tata'] 
        errExp[23]=data['Bounds']['HL-LHC']['ttxh']['err_tata'] 
        muTheo=np.concatenate((muTheo,tthmutata))



    
    



    muExp= muExp[muExp!=0]
    muTheo= muTheo[muTheo!=0]
    errExp= errExp[errExp!=0]
    


    err=(errExp**2)**0.5

    print(corr[4,0])
    # include correlations   
    A = tt.dmatrix('A')
    A.tag.test_value = np.random.rand(2, 2)
    invA = tt.nlinalg.matrix_inverse(A)
    
    f = theano.function([theano.Param(A)], invA)
    if experiments==['HL-LHC']:
        dirc= '/beegfs/desy/user/lalasfar/trilinear4tops'
        #print('yy')
        hilhccorr =np.loadtxt(dirc+"/results/correlation_matrix_CMS_HL-LHC.dat")
        cov= err.T*hilhccorr*err
        invcov=f(cov)
        #print(invcov)
        iccov = 0.5*(invcov+invcov.T)
        deltamu = (muTheo-muExp).T
        chi2d=-0.5*np.dot(np.dot(deltamu,iccov),deltamu)
        return chi2d
    else:
        cov= err.T*corr*err
        invcov=f(cov)
        iccov = 0.5*(invcov+invcov.T)
        deltamu = (muTheo-muExp).T
        chi2d=-0.5*np.dot(np.dot(deltamu,iccov),deltamu)
        return chi2d



