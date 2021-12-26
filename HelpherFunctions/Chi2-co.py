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



def mylikelihood(operator,Cqu1,CH,data,experiments=['ATLAS'],HiggsChannels=['ggf','vbf','ttxh','vh'],TopChannels='smeft_fit_quad',mode='fin',l3mode='linear',linearmu=True):
    
    """ Chi2 function in vectorissed manner
            mu0: vector of the expected poi
            mu1: vector of the observed poi
            err: uncertainties, a vector
            corr: correlation matirx
            ndf: number of defgrees of freedom
    """
    # constants
    v2=1/(2**(1/2)*data['SM']['GF'])
    v= np.sqrt(v2)
    mh=data['SM']['mh']
    pi= np.pi
    ZH= (-9*data['SM']['GF']*data['SM']['mh']**2*(-1 + (2*pi)/(3.*(3.)**.5)))/(16.*(2.)**.5*pi**2)
    xs_zh= data['SM']['xs_zh_14']if experiments ==['HL-LHC'] else  data['SM']['xs_zh_13']
    xs_wh= data['SM']['xs_wh_14']if experiments ==['HL-LHC'] else  data['SM']['xs_wh_13']



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
    #difference in cross sections from 4 fermion operators
    gamma_gaga_cqu1= data[operator]['gagaos_'+mode]* Cqu1
    gamma_gg_cqu1=data[operator]['htogg_'+mode]* Cqu1
    gamma_bb_cqu1=data[operator]['Hbb_'+mode]* Cqu1
    sigma_gg_cqu1 =data[operator]['ggFos_'+mode]* Cqu1
    sigma_ttH_cqu1= data[operator]['ttH14_'+mode]* Cqu1 if experiments== ['HL-LHC'] else data[operator]['ttH_'+mode]* Cqu1
    ttH_err = data[operator]['ttH14_delta_'+mode] if experiments== ['HL-LHC'] else data[operator]['ttH_delta_'+mode]
    cH=CH
    lambda2= 1000**2
    kl=1-2.0*CH*v**4/mh**2/1000**2
    C2 =ZH/(1-kl**2*ZH)
    lineafunc = lambda c1: -2*v**4/(mh**2*lambda2)*cH*(c1-2*ZH/(ZH-1))
    quadfunc = lambda c1:lineafunc(c1)+(4.*cH**2*v**8*ZH)/(mh**4*(-1. + ZH)**2*lambda2**2)\
    +(12.*cH**2*v**8*ZH**2)/(mh**4*(-1. + ZH)**2*lambda2**2)
      #difference in cross sections from kappa lambd
    elif l3mode=='linear':
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
        mu4leptons= quadfunc(C14l)

     #difference in branching ratios


    # total Higgs width
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
    GammaSM_WW= data['Higgs']['BRWWSM']*GammaHSM
    GammaSM_ZZ= data['Higgs']['BRZZSM']*GammaHSM
    RGamvv = (RGamzz*GammaSM_ZZ+GammaSM_WW*RGamww)/(GammaSM_WW+GammaSM_ZZ)

###########################################################################
# linearisation
###########################################################################
    if linearmu==True:
            mugaga= (1+sigma_gg_cqu1+sigma_ch_gg) +(RGamgaga)-GamHRat
            muzz =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamzz)-GamHRat
            muww =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamww)-GamHRat
            mutata =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamtata)-GamHRat
            mumm =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamtata)-GamHRat
            mubb =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGambb)-GamHRat
             # vbf
            vbfmugaga= (1+sigma_ch_vbf) +(RGamgaga)-GamHRat
            vbfmuzz =(1+sigma_ch_vbf)+(RGamzz)-GamHRat
            vbfmuww =(1+sigma_ch_vbf)+(RGamww)-GamHRat
            vbfmutata =(1+sigma_ch_vbf)+(RGamtata)-GamHRat
            vbfmumm =(1+sigma_ch_vbf)+(RGamtata)-GamHRat
            vbfmubb =(1+sigma_ch_vbf)+(RGambb)-GamHRat
             # VH
            sigma_vh = ((sigma_ch_zh*xs_zh)+(sigma_ch_wh*xs_wh))/(xs_wh+xs_zh)
            vhmugaga= (1+sigma_vh)+(RGamgaga)-GamHRat
            vhmuzz =(1+sigma_vh)+(RGamzz)-GamHRat
            vhmuww =(1+sigma_vh)+(RGamww)-GamHRat
            vhmutata =(1+sigma_vh)+(RGamtata)-GamHRat
            vhmubb =(1+sigma_vh)+(RGambb)-GamHRat
            zhmugaga= (1+sigma_ch_zh)+(RGamgaga)-GamHRat
            zhmuzz =(1+sigma_ch_zh)+(RGamzz)-GamHRat
            zhmuww =(1+sigma_ch_zh)+(RGamww)-GamHRat
            zhmutata =(1+sigma_ch_zh)+(RGamtata)-GamHRat
            zhmubb =(1+sigma_ch_zh)+(RGambb)-GamHRat
            whmugaga= (1+sigma_ch_wh)+(RGamgaga)-GamHRat
            whmuzz =(1+sigma_ch_wh)+(RGamzz)-GamHRat
            whmuww =(1+sigma_ch_wh)+(RGamww)-GamHRat
            whmutata =(1+sigma_ch_wh)+(RGamtata)-GamHRat
            whmubb =(1+sigma_ch_wh)+(RGambb)-GamHRat
             #ttH
            tthmugaga= (1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamgaga)-GamHRat
            tthmuzz =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamzz)-GamHRat
            tthmuww =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamww)-GamHRat             
            tthmuvv =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamvv)-GamHRat
            tthmutata =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamtata)-GamHRat
            tthmubb =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGambb)-GamHRat

                 
    muth1={
        'ggf':np.array([mugaga,muzz,muww,mutata,mubb,mumm]),
        'vbf': np.array([vbfmugaga,vbfmuzz,vbfmuww,vbfmutata,vbfmubb,vbfmumm]),
        'ttxh': np.array([tthmugaga,tthmuvv,0,tthmutata,tthmubb,0]),
        'ttxhhllhc': np.array([tthmugaga,tthmuzz,tthmuww,tthmubb,tthmutata,0]),
        'ttxhcms': np.array([tthmugaga,tthmuzz,0,0.,0.0,0]),
        'vh' :  np.array([vhmugaga,vhmuzz,vhmuww,0.0,vhmubb,0]),
        'zh':np.array([zhmugaga,zhmuzz,zhmuww,zhmutata,zhmubb,0]),
        'wh':np.array([whmugaga,whmuzz,whmuww,whmutata,whmubb,0])
    }

    muTheo= np.zeros(1)
    muExp= np.zeros(1)
    errExp= np.zeros(1)
    i=0
    idx_ggf=0
    idx_vh=0
    idx_vbf=0
    idx_tth=0
    for exp in experiments:
       # print(exp)
        for ch in HiggsChannels:
          #  print(ch)
            aa = data['Bounds'][exp][ch]['mu_gaga']
            err_aa=data['Bounds'][exp][ch]['err_gaga']            
            if ch=='ttxh':
                zz=data['Bounds'][exp][ch]['mu_vv']
                err_zz=data['Bounds'][exp][ch]['err_vv']
                ww = 0.0
                err_ww=0.0
                idx_tth=i
            else:
                zz = data['Bounds'][exp][ch]['mu_zz']
                err_zz=data['Bounds'][exp][ch]['err_zz']
                ww = data['Bounds'][exp][ch]['mu_ww']
                err_ww=data['Bounds'][exp][ch]['err_ww']
            if exp=='ATLAS':
                idx_vbf =i if ch=='vbf' else idx_vbf
                idx_vh =i if ch=='vh' else idx_vh
                idx_ggf =i if ch=='ggf' else idx_ggf
            tata = data['Bounds'][exp][ch]['mu_tata']
            err_tata=data['Bounds'][exp][ch]['err_tata'] 
            bb = data['Bounds'][exp][ch]['mu_bb']
            err_bb=data['Bounds'][exp][ch]['err_bb']
            if ch=='ggf' or ch=='vbf':
                mm = data['Bounds'][exp][ch]['mu_mm']
                err_mm=data['Bounds'][exp][ch]['err_mm'] 
            else:
                mm = 0
                err_mm=0
            if exp=='HL-LHC' and ch=='ttxhhllhc':
                mu=np.array([aa,zz,ww,bb,tata,mm])
                err=np.array([err_aa,err_zz,err_ww,err_bb,err_tata,err_mm])
            else:   
                mu=np.array([aa,zz,ww,tata,bb,mm])
                err=np.array([err_aa,err_zz,err_ww,err_tata,err_bb,err_mm])
               # 
            #print(err)
            muExp= np.concatenate((muExp,mu))
            errExp= np.concatenate((errExp,err))
            muth= [0,0,0,0,0,0]
            errth=[0,0,0,0,0,0]
            muth[0]= 0 if aa== 0 else  muth1[ch][0]
            muth[1]= 0 if zz== 0 else  muth1[ch][1]
            muth[2]= 0 if ww== 0 else  muth1[ch][2]
            muth[3]= 0 if mu[3]== 0 else  muth1[ch][3]
            muth[4]= 0 if mu[4]== 0 else  muth1[ch][4]
            muth[5]= 0 if mu[5]== 0 else  muth1[ch][5]
            errth[0]= 0 if aa== 0 else  errth1[ch][0]
            errth[1]= 0 if zz== 0 else errth1[ch][1]
            errth[2]= 0 if ww== 0 else  errth1[ch][2]
            errth[3]= 0 if err[3]== 0 else  errth1[ch][3]
            errth[4]= 0 if err[4]== 0 else  errth1[ch][4]
            errth[5]= 0 if err[5]== 0 else  errth1[ch][5]
            if exp=='CMS' and ch=='ttxh':
                errTheo=np.concatenate((errTheo,errth1['ttxhcms']))
                muTheo=np.concatenate((muTheo, muth1['ttxhcms']))
            else:
                errTheo=np.concatenate((errTheo,errth))
                muTheo=np.concatenate((muTheo, muth))
            i=i+1
            #print(muTheo)
            

        
 
 
    muExp= muExp[muExp!=0]
    muTheo= muTheo[muTheo!=0]
    errExp= errExp[errExp!=0]
    errTheo= errTheo[errTheo!=0]

    if TopChannels!=None:
        muTheo=np.concatenate((muTheo,np.array([Cqu1])))
        muExp=np.concatenate((muExp, [data['Bounds'][TopChannels][operator]]))
        errExp=np.concatenate((errExp, [data['Bounds'][TopChannels]['delta_'+operator]]))

    if TopChannels!=None:
        errTheo=np.concatenate((errTheo, [0.0]))

    
   
    num_of_obs=muExp.shape[0]

    corr = np.identity(num_of_obs, dtype = float)
    corrd = np.identity(num_of_obs, dtype = float)


    #  correlations
    if experiments !=['HL-LHC']:
        #print('ss')
        corr[idx_ggf+0,idx_vbf+0] = data['Bounds']['ATLAS']['corr_ggfvbf_gaga']
        corr[idx_vbf+0,idx_ggf+0] =data['Bounds']['ATLAS']['corr_ggfvbf_gaga']
        corr[idx_ggf+1,idx_vbf+1] = data['Bounds']['ATLAS']['corr_ggfvbf_zz']
        corr[idx_vbf+1,idx_ggf+1] =data['Bounds']['ATLAS']['corr_ggfvbf_zz']
        corr[idx_ggf+2,idx_vbf+2] = data['Bounds']['ATLAS']['corr_ggfvbf_ww']
        corr[idx_vbf+2,idx_ggf+2] =data['Bounds']['ATLAS']['corr_ggfvbf_ww']
        corr[idx_ggf+3,idx_vbf+3] = data['Bounds']['ATLAS']['corr_ggfvbf_tata']
        corr[idx_vbf+3,idx_ggf+3] =data['Bounds']['ATLAS']['corr_ggfvbf_tata']
        corr[idx_ggf+1,idx_vh+1] = data['Bounds']['ATLAS']['corr_ggf_vh_zz']
        corr[idx_vh+1,idx_ggf+1] =data['Bounds']['ATLAS']['corr_ggf_vh_zz']
        corr[idx_tth+1,idx_tth+2] = data['Bounds']['ATLAS']['corr_tth_tatazz']
        corr[idx_tth+2,idx_tth+1] =data['Bounds']['ATLAS']['corr_tth_tatazz']
        
    
 

    ##### compute the correlation part ( keep only quadratic terms)
    
    ndf= 2.

    err=(errExp**2)**0.5

    # include correlations   
    A = tt.dmatrix('A')
    A.tag.test_value = np.random.rand(2, 2)
    invA = tt.nlinalg.matrix_inverse(A)
    
    f = theano.function([theano.Param(A)], invA)
    if experiments==['HL-LHC']:
        dirc= '/beegfs/desy/user/lalasfar/trilinear4tops'
        hilhccorr =np.loadtxt(dirc+"/results/correlation_matrix_CMS_HL-LHC.dat")
        cov= err.T*hilhccorr*err
        invcov=f(cov)
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
        
    




