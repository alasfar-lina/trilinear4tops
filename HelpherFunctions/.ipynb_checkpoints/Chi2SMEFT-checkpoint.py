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



def mylikelihoodAV(CH,CHG,Cbox,CHDD,CHW,CHB,CHWB,CHl1,CHl3,CHe,CHq1,CHq3,CHu,CHd,
                   CtaH,CtH,CbH,
                   CG,CtG,CtW,CtB,
                   Cll1,Cqt1,Cqt8,Cqtqb1,Cqtqb8,cqq3,cqq31,cqq1,cqq11,cuu,cuu1,cud8,cud1,cqu8,cqu1,cqd8,cqd1,
                   data,collider='Run-II',mode='fin',l3mode='linear',operator='Cqq3'):
    
    # constants
    lambda2= 1000**2
    v2=1/(2**(1/2)*data['SM']['GF'])
    v= np.sqrt(v2)
    mh=data['SM']['mh']
    pi= np.pi
    ZH= (-9*data['SM']['GF']*data['SM']['mh']**2*(-1 + (2*pi)/(3.*(3.)**.5)))/(16.*(2.)**.5*pi**2)
    xs_zh= data['SM']['xs_zh_14']if collider =='HL-LHC' else  data['SM']['xs_zh_13']
    xs_wh= data['SM']['xs_wh_14']if collider =='HL-LHC' else  data['SM']['xs_wh_13']

############# LO modifications #####################
####################################################
    ggf = 35.*CHG
    #
    VBF = (0.1213* Cbox - 0.0107* CHDD - 0.008 *CHW + 
     0.0313* CHWB - 0.364*CHl3 + 0.0043* CHq1 - 
     0.212* CHq3 - 0.0108 *CHu + 0.0038*CHd + 
     0.182*Cll1)
    #
    VH = (0.120 *Cbox - 0.0071 *CHDD+ 0.623* CHW + 
   0.0215* CHB + 0.098* CHWB - 0.360* CHl3 - 
   0.026 *CHq1 + 1.86 *CHq3 + 0.135* CHu- 
   0.0506* CHd + 0.181* Cll1)
    #
    ttH = (0.133*CG + 0.1182* Cbox - 0.0296* CHDD + 0.532*CHG+ 
   0.0120* CHW - 0.1152* np.abs(CtH) - 0.790*np.abs(CtG) - 
   0.0111* np.abs(CtW) - 0.0017* np.abs(CtB) - 0.1320* CHl3 + 
   0.0146 *CHq3 + 0.0660* Cll1 + 0.0218*cqq1 + 0.1601*cqq11 + 
   0.0263*cqq3 + 0.388*cqq31 + 0.0114*cuu + 0.1681*cuu1 - 
   0.0018*cud1 + 0.0265*cud8 + 0.007*cqu1 + 0.1087*cqu8 - 
   0.0011*cqd1 + 0.0266*cqd8)
    #
    Htoaa = (-13.996*CHW - 48.809*CHB+ 26.144* CHWB )
    #
    HtoZZ = (0.119*Cbox + 0.005*CHDD - 
   0.296*CHW- .197*CHB+ .296*CHWB + 
   0.126* CHl1 - 0.234*CHl3- 0.101* CHe + 0.181*Cll1)
    #
    HtoWW = (0.121* Cbox - 0.031*CHDD - 
   0.095* CHW + .006*CHB + .002* CHWB - 
   0.228*CHl3- 0.004* CHe + 0.181*Cll1)
    #
    Htotata = (0.121* Cbox - 0.030* CHDD - 0.121* np.abs(CtaH) - 
   0.121*CHl3 + 0.061*Cll1)
    #
    Htobb =  (0.121*Cbox - 0.030*CHDD - 0.121* np.abs(CbH) - 
   0.121* CHl3+ 0.061*Cll1)
    


############# 4F modifications #####################
######################################################

    gamma_gaga_cqu1= data['Cqt1']['gagaos_'+mode]* Cqt1+data['Cqt8']['gagaos_'+mode]* Cqt8+data['Cqtqb1']['gagaos_'+mode]* Cqtqb1+data['Cqtqb8']['gagaos_'+mode]*Cqtqb8+Htoaa
    #+data['Cqq1']['gagaos']*Cqq1+data['Cqq8']['gagaos']*Cqq8
    #
    gamma_gg_cqu1=data['Cqt1']['htogg_'+mode]* Cqt1+data['Cqt8']['htogg_'+mode]* Cqt8+data['Cqtqb1']['htogg_'+mode]* Cqtqb1+data['Cqtqb8']['htogg_'+mode]*Cqtqb8+ggf 
    #+data['Ctt1']['ggFos']*Ctt1+data['Cqq1']['ggFos']*Cqq1+data['Cqq8']['ggFos']*Cqq8
    #
    gamma_bb_cqu1=data['Cqt1']['Hbb_'+mode]* Cqt1+data['Cqt8']['Hbb_'+mode]* Cqt8+data['Cqtqb1']['Hbb_'+mode]* Cqtqb1+data['Cqtqb8']['Hbb_'+mode]*Cqtqb8+Htobb
    #+data['Ctt1']['Hbb']*Ctt1+data['Cqq1']['Hbb']*Cqq1+data['Cqq8']['Hbb']*Cqq8
    sigma_gg_cqu1 =data['Cqt1']['ggFos_'+mode]* Cqt1+data['Cqt8']['ggFos_'+mode]* Cqt8+data['Cqtqb1']['ggFos_'+mode]* Cqtqb1+data['Cqtqb8']['ggFos_'+mode]*Cqtqb8+ggf
    #+data['Ctt1']['ggFos']*Ctt1+data['Cqq1']['ggFos']*Cqq1+data['Cqq8']['ggFos']*Cqq8
    #
    #
    if collider== 'HL-LHC':
        sigma_ttH_cqu1=data['Cqt1']['ttH14_'+mode]* Cqt1+data['Cqt8']['ttH14_'+mode]* Cqt8+data['Cqtqb1']['ttH14_'+mode]* Cqtqb1+data['Cqtqb8']['ttH14_'+mode]*Cqtqb8+ttH
    else:
        sigma_ttH_cqu1=data['Cqt1']['ttH_'+mode]* Cqt1+data['Cqt8']['ttH_'+mode]* Cqt8+data['Cqtqb1']['ttH_'+mode]* Cqtqb1+data['Cqtqb8']['ttH_'+mode]*Cqtqb8+ttH
    #+data['Ctt1']['ttH']*Ctt1+data['Cqq1']['ttH']*Cqq1+data['Cqq8']['ttH']*Cqq8
    
############# Cphi modifications #####################
######################################################  
    #theoretical input
    C1gaga=data['kl']['gaga']
    C1gg = data['kl']['ggF']
    C1zz =data['kl']['zz']
    C1ww=data['kl']['ww']
    C1ff=data['kl']['ff']
    C1VBF=data['kl']['VBF']
    C1ZH=data['kl']['ZH14']  if collider== 'HL-LHC' else data['kl']['ZH'] 
    C1WH=data['kl']['WH']
    C1ttH= data['kl']['ttH14'] if collider== 'HL-LHC' else data['kl']['ttH'] 
    C14l=data['kl']['hto4l']
    d_gamma_tot_ch = data['kl']['tot_gamma']
    ############
    cH=CH
    lineafunc = lambda c1: -2*v**4/(mh**2*lambda2)*cH*(c1-2*ZH/(ZH-1))
    quadfunc = lambda c1:lineafunc(c1)+(4.*cH**2*v**8*ZH)/(mh**4*(-1. + ZH)**2*lambda2**2)\
    +(12.*cH**2*v**8*ZH**2)/(mh**4*(-1. + ZH)**2*lambda2**2)

    if l3mode=='linear':
        sigma_ch_gg= lineafunc(C1gg)
        sigma_ch_vbf= lineafunc(C1VBF)+VBF
        sigma_ch_zh= lineafunc(C1ZH)
        sigma_ch_wh=lineafunc(C1WH)
        sigma_ch_tth= lineafunc(C1ttH)
        ###
        RGamgaga =lineafunc(C1gaga)+gamma_gaga_cqu1
        RGamgg =lineafunc(C1gg)+gamma_gg_cqu1
        RGamww =lineafunc(C1ww)+HtoWW
        RGamzz =lineafunc(C1zz)+HtoZZ
        RGambb =lineafunc(C1ff)+gamma_bb_cqu1
        RGamtata =lineafunc(C1ff)+Htotata
        RGamff =lineafunc(C1ff)
        mu4leptons= lineafunc(C14l)
    elif l3mode=='quadratic':
        sigma_ch_gg= quadfunc(C1gg)
        sigma_ch_vbf= quadfunc(C1VBF)+VBF
        sigma_ch_zh= quadfunc(C1ZH)
        sigma_ch_wh=quadfunc(C1WH)
        sigma_ch_tth= quadfunc(C1ttH)
        ###
        RGamgaga =quadfunc(C1gaga)+gamma_gaga_cqu1
        RGamgg =quadfunc(C1gg)+gamma_gg_cqu1
        RGamww =quadfunc(C1ww)+HtoWW
        RGamzz =quadfunc(C1zz)+HtoZZ
        RGambb =quadfunc(C1ff)+gamma_bb_cqu1
        RGamtata =quadfunc(C1ff)+Htotata
        RGamff =quadfunc(C1ff)
        mu4leptons= quadfunc(C14l) # not used 
    else:
         raise NameError('Incorrect mode selected')

############## total Higgs width #####################
######################################################  

    GammaSM_WW= data['Higgs']['BRWWSM']*data['Higgs']['width']
    GammaSM_ZZ= data['Higgs']['BRZZSM']*data['Higgs']['width']
    ###
    GammSM_bb =data['Higgs']['BRbbSM']*data['Higgs']['width']
    RGamvv = (RGamzz*GammaSM_ZZ+GammaSM_WW*RGamww)/(GammaSM_WW+GammaSM_ZZ)
    
    GamHRat = (RGambb*\
    data['Higgs']['BRbbSM'] +RGamff*\
    data['Higgs']['BRccSM'] + RGamtata*\
    data['Higgs']['BRTauTauSM'] + RGamff*\
    data['Higgs']['BRMuMuSM'] + RGamww*\
    data['Higgs']['BRWWSM'] + RGamzz*\
    data['Higgs']['BRZZSM'] + RGamgaga*\
    data['Higgs']['BRgagaSM'] + RGamgg*data['Higgs']['BRggSM'])
    #/(data['Higgs']['BRbbSM'] +
    #data['Higgs']['BRccSM'] + data['Higgs']['BRTauTauSM'] + data['Higgs']['BRMuMuSM'] + data['Higgs']['BRWWSM'] + data['Higgs']['BRZZSM'] +
    #data['Higgs']['BRgagaSM'] + data['Higgs']['BRggSM'])

    GammaHSM= (data['Higgs']['BRbbSM'] +
    data['Higgs']['BRccSM'] + data['Higgs']['BRTauTauSM'] + data['Higgs']['BRMuMuSM'] + data['Higgs']['BRWWSM'] + data['Higgs']['BRZZSM'] +
    data['Higgs']['BRgagaSM'] + data['Higgs']['BRggSM'])

###########################################################################
# linearisation of the signal strength
###########################################################################
    sigma_vh = ((sigma_ch_zh*xs_zh)+(sigma_ch_wh*xs_wh))/(xs_wh+xs_zh)+VH
    muth={
    
        'ggf':{
            'gaga':(1+sigma_gg_cqu1+sigma_ch_gg)+(RGamgaga)-GamHRat,
            'zz': (1+sigma_ch_gg+sigma_gg_cqu1)+(RGamzz)-GamHRat,
            'ww':(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamww)-GamHRat,
            'tata':(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamtata)-GamHRat,
            'bb':(1+sigma_ch_gg+sigma_gg_cqu1)+(RGambb)-GamHRat,
            'mm':(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamtata)-GamHRat
        },
        'vbf':{
            'gaga':(1+sigma_ch_vbf) +(RGamgaga)-GamHRat,
            'zz':(1+sigma_ch_vbf)+(RGamzz)-GamHRat,
            'ww':(1+sigma_ch_vbf)+(RGamww)-GamHRat,
            'tata':(1+sigma_ch_vbf)+(RGamtata)-GamHRat,
            'bb':(1+sigma_ch_vbf)+(RGambb)-GamHRat,
            'mm':(1+sigma_ch_vbf)+(RGamtata)-GamHRat
        },
        'vh':{
            'gaga':(1+sigma_vh)+(RGamgaga)-GamHRat,
            'zz':(1+sigma_vh)+(RGamzz)-GamHRat,
            'ww':(1+sigma_vh)+(RGamww)-GamHRat,
            'tata':(1+sigma_vh)+(RGamtata)-GamHRat,
            'bb':(1+sigma_vh)+(RGambb)-GamHRat
        },
        'zh':{
            'gaga':(1+sigma_ch_zh)+(RGamgaga)-GamHRat,
            'zz':(1+sigma_ch_zh)+(RGamzz)-GamHRat,
            'ww':(1+sigma_ch_zh)+(RGamww)-GamHRat,
            'tata':(1+sigma_ch_zh)+(RGamtata)-GamHRat,
            'bb':(1+sigma_ch_zh)+(RGambb)-GamHRat
        },
        'wh':{
            'gaga':(1+sigma_ch_wh)+(RGamgaga)-GamHRat,
            'zz':(1+sigma_ch_wh)+(RGamzz)-GamHRat,
            'ww':(1+sigma_ch_wh)+(RGamww)-GamHRat,
            'tata':(1+sigma_ch_wh)+(RGamtata)-GamHRat,
            'bb':(1+sigma_ch_wh)+(RGambb)-GamHRat
        },
        'ttxh':{
            'gaga':(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamgaga)-GamHRat,
            'zz':(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamzz)-GamHRat,
            'ww':(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamww)-GamHRat,
            'vv':(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamvv)-GamHRat,
            'tata':(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamtata)-GamHRat,
            'bb':(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGambb)-GamHRat
        }
    }



###########################################################################
# 
#  Matching to experimental input
#
###########################################################################    

    num_of_obs=16
    corr = np.identity(num_of_obs, dtype = float)
    #symmetrise errors
    def SymMu (dat,exp,ch,decay):
        mu= dat['Bounds'][exp][ch]['mu_'+decay]
        up=dat['Bounds'][exp][ch]['up_err_'+decay]
        low=dat['Bounds'][exp][ch]['low_err_'+decay]
        return mu+0.5*(up-low)
    #########################
    def SymErr (dat,exp,ch,decay):
        up=dat['Bounds'][exp][ch]['up_err_'+decay]
        low=dat['Bounds'][exp][ch]['low_err_'+decay]
        return 0.5*(up+low)

    
###############################################################
#           Run-II LHC
 ############################################################### 
    muTheo= np.zeros(1)
    muExp= np.zeros(1)
    errExp= np.zeros(1)

            
            
            ### the ordr here is important 
    experiments ={
        'Run-II':['ATLAS','CMS'],
        'HL-LHC':['HL-LHC']
        }        
    channels ={
        'ATLAS' :['ggf','vbf','ttxh','vh'],
        'CMS':['ggf','vbf','ttxh','vh','wh','zh'],
        'HL-LHC':['ggf','vbf','wh','zh','ttxh']
            }
    decays={
        'ATLAS':{
            'ggf': ['gaga','zz','ww','tata'],
            'vbf':['gaga','zz','ww','tata','bb'],
            'ttxh':['gaga','vv','tata','bb'],
            'vh':['gaga','zz','bb']
            },
        'CMS':{
            'ggf': ['gaga','zz','ww','tata','bb'
                    ,'mm'
                   ],
            'vbf':['gaga','zz','ww','tata'
                   ,'mm'
                  ],
            'ttxh':['gaga'
                   ,'zz','ww','tata','bb'
                   ],
            'vh':['gaga','zz','ww'],
            'wh':['tata','bb'],
            'zh':['tata','bb']
            },
        'HL-LHC':{
            'ggf':['gaga','zz','ww','tata','bb','mm'],
            'vbf':['gaga','zz','ww','tata','mm'],
            'wh':['gaga','zz','ww','bb'],
            'zh':['gaga','zz','ww','bb'],
            'ttxh':['gaga','zz','ww','bb','tata']
        }
        }
    for exp in experiments[collider]:
        for ch in channels[exp]:
            for dec in decays[exp][ch]:
                muExp= np.concatenate((muExp,[SymMu(data,exp,ch,dec)]))
                errExp= np.concatenate((errExp,[SymErr(data,exp,ch,dec)]))
                muTheo=np.concatenate((muTheo,[muth[ch][dec]]))
                #print(exp,ch,dec,SymMu(data,exp,ch,dec),'\pm',SymErr(data,exp,ch,dec))

                
    
    
    
    
    

          #setting up correlations 
    muExp= muExp[muExp!=0]
    muTheo= muTheo[muTheo!=0]
    errExp= errExp[errExp!=0]
    num_of_obs=muExp.shape[0]
    #print(muTheo)

    corr = np.identity(num_of_obs, dtype = float)
    if collider=='Run-II':
            # ATLAS
        ### gamma gamma between ggf and vbf
        corr[0,4]= data['Bounds']['ATLAS']['corr_ggfvbf_gaga']      
        corr[4,0]=corr[0,4]    
        ### ZZ between ggf and vbf
        corr[1,5]= data['Bounds']['ATLAS']['corr_ggfvbf_zz']   
        corr[5,1]=corr[1,5]   
        ### tata between ggf and vbf
        corr[3,7]= data['Bounds']['ATLAS']['corr_ggfvbf_tata']
        corr[7,3]=corr[3,7]   
        ### ZZ between ggf and vh
        corr[1,14]=data['Bounds']['ATLAS']['corr_ggf_vh_zz']               
        corr[14,1]=corr[1,14]   
        ### t t h between ta ta and VV
        corr[11,10]=data['Bounds']['ATLAS']['corr_tth_tatazz']               
        corr[10,11]=corr[11,10]     

    
    ####
    err=errExp
    
    
    


    
    # include correlations   
    A = tt.dmatrix('A')
    A.tag.test_value = np.random.rand(2, 2)
    invA = tt.nlinalg.matrix_inverse(A)
    
    f = theano.function([theano.Param(A)], invA)
    if collider=='HL-LHC':
        dirc= '/beegfs/desy/user/lalasfar/trilinear4tops'
        #print('yy')
        hilhccorr =np.loadtxt(dirc+"/results/correlation_matrix_CMS_HL-LHC.dat")
        outer_err=np.outer(err, err)
        cov= hilhccorr*outer_err
        invcov=f(cov)
        #print(invcov)
        iccov = 0.5*(invcov+invcov.T)
        deltamu = (muTheo-muExp).T
        chi2d=-0.5*np.dot(np.dot(deltamu,iccov),deltamu)
        return chi2d
    else:
        outer_err=np.outer(err, err)
        cov= corr*outer_err
        invcov=f(cov)
        iccov = 0.5*(invcov+invcov.T)
        deltamu = (muTheo-muExp).T
        chi2d=-0.5*np.dot(np.dot(deltamu,iccov),deltamu)
        return chi2d






