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



def mylikelihoodAV(Cqu1,Cqt8,Cqtqb1,Cqtqb8,CH,data,experiments=['ATLAS','CMS'],HiggsChannels=['ggf','vbf','ttxh','vh'],TopChannels=None,linearl3=True,linearmu=True):
    
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



    #theoretical input
    C1gaga=data['kl']['gaga']
    C1gg = data['kl']['ggF']
    C1zz =data['kl']['zz']
    C1ww=data['kl']['ww']
    C1ff=data['kl']['ff']
    C1VBF=data['kl']['VBF']
    C1ZH=data['kl']['ZH']
    C1WH=data['kl']['WH']
    C1ttH=data['kl']['ttH']
    d_gamma_tot_ch = data['kl']['tot_gamma']
    #difference in cross sections from 4 fermion operators
    gamma_gaga_cqu1= data['Cqt1']['gagaos']* Cqu1+data['Cqt8']['gagaos']* Cqt8+data['Cqtqb1']['gagaos']* Cqtqb1+data['Cqtqb8']['gagaos']*Cqtqb8
    #+data['Ctt1']['gagaos']*Ctt1+data['Cqq1']['gagaos']*Cqq1+data['Cqq8']['gagaos']*Cqq8
    #
    gamma_gg_cqu1=data['Cqt1']['ggFos']* Cqu1+data['Cqt8']['ggFos']* Cqt8+data['Cqtqb1']['ggFos']* Cqtqb1+data['Cqtqb8']['ggFos']*Cqtqb8
    #+data['Ctt1']['ggFos']*Ctt1+data['Cqq1']['ggFos']*Cqq1+data['Cqq8']['ggFos']*Cqq8
    #
    gamma_bb_cqu1=data['Cqt1']['Hbb']* Cqu1+data['Cqt8']['Hbb']* Cqt8+data['Cqtqb1']['Hbb']* Cqtqb1+data['Cqtqb8']['Hbb']*Cqtqb8
    #+data['Ctt1']['Hbb']*Ctt1+data['Cqq1']['Hbb']*Cqq1+data['Cqq8']['Hbb']*Cqq8
    sigma_gg_cqu1 =gamma_gg_cqu1
    #
    sigma_ttH_cqu1=data['Cqt1']['ttH']* Cqu1+data['Cqt8']['ttH']* Cqt8+data['Cqtqb1']['ttH']* Cqtqb1+data['Cqtqb8']['ttH']*Cqtqb8
    #+data['Ctt1']['ttH']*Ctt1+data['Cqq1']['ttH']*Cqq1+data['Cqq8']['ttH']*Cqq8
    cH=CH
    cH=CH
    kl=1-2.0*CH*v**4/mh**2/1000**2
    C2 =ZH/(1-kl**2*ZH) 
      #difference in cross sections from kappa lambd
    if linearl3==False:
        sigma_ch_gg= (kl-1)*C1gg+(kl**2-1)*C2
        sigma_ch_vbf= (kl-1)*C1VBF+(kl**2-1)*C2
        sigma_ch_zh= (kl-1)*C1ZH+(kl**2-1)*C2
        sigma_ch_wh= (kl-1)*C1WH+(kl**2-1)*C2
        sigma_ch_tth= (kl-1)*C1ttH+(kl**2-1)*C2
        RGamgaga =(1+ (kl-1)*C1gaga+(kl**2-1)*C2)+gamma_gaga_cqu1
        RGamgg =(1+ (kl-1)*C1gg+(kl**2-1)*C2)+gamma_gg_cqu1
        RGamww =(1+ (kl-1)*C1ww+(kl**2-1)*C2)
        RGamzz =(1+ (kl-1)*C1zz+(kl**2-1)*C2)
        RGambb =(1+ (kl-1)*C1ff+(kl**2-1)*C2)+gamma_bb_cqu1
        RGamtata =(1+ (kl-1)*C1ff+(kl**2-1)*C2)
        RGamff =(1+ (kl-1)*C1ff+(kl**2-1)*C2)
    else:
        sigma_ch_gg= (-2*cH*v**2*(C1gg+2*v**2*ZH))/(mh**2*1000**2)
        sigma_ch_vbf= (-2*cH*v**2*(C1VBF+2*v**2*ZH))/(mh**2*1000**2)
        sigma_ch_zh= (-2*cH*v**2*(C1ZH+2*v**2*ZH))/(mh**2*1000**2)
        sigma_ch_wh=(-2*cH*v**2*(C1WH+2*v**2*ZH))/(mh**2*1000**2)
        sigma_ch_tth= (-2*cH*v**2*(C1ttH+2*v**2*ZH))/(mh**2*1000**2)
        ###
        RGamgaga =1+(-2*cH*v**2*(C1gaga+2*v**2*ZH))/(mh**2*1000**2)+gamma_gaga_cqu1
        RGamgg =1+(-2*cH*v**2*(C1gg+2*v**2*ZH))/(mh**2*1000**2)+gamma_gg_cqu1
        RGamww =1+(-2*cH*v**2*(C1ww+2*v**2*ZH))/(mh**2*1000**2)
        RGamzz =1+(-2*cH*v**2*(C1zz+2*v**2*ZH))/(mh**2*1000**2)
        RGambb =1+(-2*cH*v**2*(C1ff+2*v**2*ZH))/(mh**2*1000**2)+gamma_bb_cqu1
        RGamtata =1+(-2*cH*v**2*(C1ff+2*v**2*ZH))/(mh**2*1000**2)
        RGamff =1+(-2*cH*v**2*(C1ff+2*v**2*ZH))/(mh**2*1000**2)

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

###########################################################################
# linearisation
###########################################################################
    if linearmu==True:
            GamHRat=GamHRat-1
            mugaga= (1+sigma_gg_cqu1+sigma_ch_gg) +(RGamgaga-1)-GamHRat
            muzz =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamzz-1)-GamHRat
            muww =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamww-1)-GamHRat
            mutata =(1+sigma_ch_gg+sigma_gg_cqu1)+(RGamtata-1)-GamHRat
            mubb =(1+sigma_ch_gg)+(RGambb-1)-GamHRat
             # vbf
            vbfmugaga= (1+sigma_ch_vbf) +(RGamgaga-1)-GamHRat
            vbfmuzz =(1+sigma_ch_vbf)+(RGamzz-1)-GamHRat
            vbfmuww =(1+sigma_ch_vbf)+(RGamww-1)-GamHRat
            vbfmutata =(1+sigma_ch_vbf)+(RGamtata-1)-GamHRat
            vbfmubb =(1+sigma_ch_vbf)+(RGambb-1)-GamHRat
             # VH
            vhmugaga= (1+sigma_ch_zh+sigma_ch_wh)+(RGamgaga-1)-GamHRat
            vhmuzz =(1+sigma_ch_zh+sigma_ch_wh)+(RGamzz-1)-GamHRat
            vhmuww =(1+sigma_ch_zh+sigma_ch_wh)+(RGamww-1)-GamHRat
            vhmubb =(1+sigma_ch_zh+sigma_ch_wh)+(RGambb-1)-GamHRat
            zhmugaga= (1+sigma_ch_zh)+(RGamgaga-1)-GamHRat
            zhmuzz =(1+sigma_ch_zh)+(RGamzz-1)-GamHRat
            zhmuww =(1+sigma_ch_zh)+(RGamww-1)-GamHRat
            zhmubb =(1+sigma_ch_zh)+(RGambb-1)-GamHRat
            whmugaga= (1+sigma_ch_wh)+(RGamgaga-1)-GamHRat
            whmuzz =(1+sigma_ch_wh)+(RGamzz-1)-GamHRat
            whmuww =(1+sigma_ch_wh)+(RGamww-1)-GamHRat
            whmubb =(1+sigma_ch_wh)+(RGambb-1)-GamHRat
             #ttH
            tthmugaga= (1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamgaga-1)-GamHRat
            tthmuvv =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamzz-1+RGamww-1)-GamHRat
            tthmutata =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGamtata-1)-GamHRat
            tthmubb =(1+sigma_ch_tth+sigma_ttH_cqu1)+(RGambb-1)-GamHRat
            tth4l=tthmuvv+tthmutata-1
    if linearmu==False:
              #GluonFusion
            mugaga= (1+sigma_gg_cqu1+sigma_ch_gg)*(RGamgaga)/GamHRat
            muzz =(1+sigma_ch_gg+sigma_gg_cqu1)*(RGamzz)/GamHRat
            muww =(1+sigma_ch_gg+sigma_gg_cqu1)*(RGamww)/GamHRat
            mutata =(1+sigma_ch_gg+sigma_gg_cqu1)*(RGamtata)/GamHRat
            mubb =(1+sigma_ch_gg+sigma_gg_cqu1)*(RGambb)/GamHRat
             # vbf
            vbfmugaga= (1+sigma_ch_vbf) *(RGamgaga)/GamHRat
            vbfmuzz =(1+sigma_ch_vbf)*(RGamzz)/GamHRat
            vbfmuww =(1+sigma_ch_vbf)*RGamww/GamHRat
            vbfmutata =(1+sigma_ch_vbf)*RGamtata/GamHRat
            vbfmubb =(1+sigma_ch_vbf)*RGambb/GamHRat
             # VH
            vhmugaga= (1+sigma_ch_zh+sigma_ch_wh)*RGamgaga/GamHRat
            vhmuzz =(1+sigma_ch_zh+sigma_ch_wh)*RGamzz/GamHRat
            vhmuww =(1+sigma_ch_zh+sigma_ch_wh)*RGamww/GamHRat
            vhmubb =(1+sigma_ch_zh+sigma_ch_wh)*RGambb/GamHRat
            zhmugaga= (1+sigma_ch_zh)*RGamgaga/GamHRat
            zhmuzz =(1+sigma_ch_zh)*RGamzz/GamHRat
            zhmuww =(1+sigma_ch_zh)*RGamww/GamHRat
            zhmubb =(1+sigma_ch_zh)*RGambb/GamHRat
            whmugaga= (1+sigma_ch_wh)*RGamgaga/GamHRat
            whmuzz =(1+sigma_ch_wh)*RGamzz/GamHRat
            whmuww =(1+sigma_ch_wh)*RGamww/GamHRat
            whmubb =(1+sigma_ch_wh)*RGambb/GamHRat            
             #ttH
            tthmugaga= (1+sigma_ch_tth+sigma_ttH_cqu1)*RGamgaga/GamHRat
            tthmuvv =(1+sigma_ch_tth+sigma_ttH_cqu1)*(RGamww+RGamzz)/GamHRat
            tthmutata =(1+sigma_ch_tth+sigma_ttH_cqu1)*RGamtata/GamHRat
            tthmubb =(1+sigma_ch_tth+sigma_ttH_cqu1)*RGambb/GamHRat
            tth4l=tthmuvv+tthmutata-1

###########################################################################
     #theoretical errors
    delta_ch_gaga =((1./(2.)**.5*(1)**2 * C1gaga *ZH)**2)**.5
    delta_ch_zz =((1./(2.)**.5 *(1)**2* C1zz *ZH)**2)**.5
    delta_ch_ww =((1./(2.)**.5*(1)**2 *C1ww *ZH)**2)**.5
    delta_ch_gg =((1./(2.)**.5 *(1)**2* C1gg *ZH)**2)**.5
    delta_ch_vbf =((1./(2.)**.5*(1)**2 * C1VBF *ZH)**2)**.5
    delta_ch_zh =((1/(2.)**.5 *(1)**2* C1ZH *ZH)**2)**.5
    delta_ch_wh =((1/(2.)**.5*(1)**2 *C1WH *ZH)**2)**.5
    delta_ch_tth =((1/(2)**.5 *(1)**2 * C1ttH *ZH)**2)**.5

      # merge the WH and ZH channels

    delta_ch_vh = (delta_ch_zh**2+delta_ch_wh**2 )**.5
    delta_ch_vhdecay = (delta_ch_ww**2+delta_ch_zz**2 )**.5
    

      # gluon fusion theoretical error
    errggfTheo = np.array([(delta_ch_gg**2+delta_ch_gaga**2)**.5,\
    (delta_ch_gg**2+delta_ch_zz**2)**.5,\
    (delta_ch_gg**2+delta_ch_ww**2)**.5,\
    (delta_ch_gg**2)**.5,\
    (delta_ch_gg**2)**.5
        ])
        # VBF theoretical error
    errvbfTheo = np.array([(delta_ch_vbf**2+delta_ch_gaga**2)**.5,\
   (delta_ch_vbf**2+delta_ch_zz**2)**.5,\
   (delta_ch_vbf**2+delta_ch_ww**2)**.5,\
    (delta_ch_vbf**2)**.5,\
    (delta_ch_vbf**2)**.5\
        ])
    errvhTheo = np.array([(delta_ch_vh**2+delta_ch_gaga**2)**.5,\
    (delta_ch_vh**2+delta_ch_zz**2)**.5,\
    (delta_ch_vh**2+delta_ch_ww**2)**.5,(delta_ch_vh**2)**.5,\
    (delta_ch_vh**2)**.5\
        ])
    errwhTheo = np.array([(delta_ch_wh**2+delta_ch_gaga**2)**.5,\
    (delta_ch_wh**2+delta_ch_zz**2)**.5,\
    (delta_ch_wh**2+delta_ch_ww**2)**.5,(delta_ch_vh**2)**.5,\
    (delta_ch_wh**2)**.5\
        ])
    errzhTheo = np.array([(delta_ch_zh**2+delta_ch_gaga**2)**.5,\
    (delta_ch_zh**2+delta_ch_zz**2)**.5,\
    (delta_ch_zh**2+delta_ch_ww**2)**.5,(delta_ch_vh**2)**.5,\
    (delta_ch_zh**2)**.5\
        ])
        
    errtthTheo = np.array([(delta_ch_tth**2+delta_ch_gaga**2)**.5,\
    (delta_ch_tth**2+delta_ch_vhdecay**2)**.5,\
                           0.,\
    (delta_ch_tth**2)**.5,\
    (delta_ch_tth**2)**.5\
        ])
    errtthTheocms = np.array([(delta_ch_tth**2)**.5,\
    (delta_ch_tth**2+delta_ch_vhdecay**2)**.5,\
                           0.,0.,0.,\
        ])
    muth1={
        'ggf':np.array([mugaga,muzz,muww,mutata,mubb]),
        'vbf': np.array([vbfmugaga,vbfmuzz,vbfmuww,vbfmutata,vbfmubb]),
        'ttxh': np.array([tthmugaga,tthmuvv,0,tthmutata,tthmubb]),
        'ttxhcms': np.array([tthmugaga,tth4l,0,0.,0.0]),
        'vh' :  np.array([vhmugaga,vhmuzz,vhmuww,vhmuww,vhmubb]),
        'zh':np.array([zhmugaga,zhmuzz,zhmuww,zhmuww,zhmubb]),
        'wh':np.array([whmugaga,whmuzz,whmuww,whmuww,whmubb])
    }
    errth1={
        'ggf':errggfTheo,
        'vbf':errvbfTheo,
        'ttxh':errtthTheo,
       'ttxhcms':errtthTheocms,
        'vh' : errvhTheo,
        'zh':errzhTheo,
        'wh':errwhTheo
    }
    muTheo= np.zeros(1)
    muExp= np.zeros(1)
    errExp= np.zeros(1)
    errTheo= np.zeros(1)
    i=0
    idx_ggf=0
    idx_vh=0
    idx_vbf=0
    idx_tth=0
    for exp in experiments:
        for ch in HiggsChannels:
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
            mu=np.array([aa,zz,ww,tata,bb])
               #
            err=np.array([err_aa,err_zz,err_ww,err_tata,err_bb])
            muExp= np.concatenate((muExp,mu))
            errExp= np.concatenate((errExp,err))
            muth= [0,0,0,0,0]
            errth=[0,0,0,0,0]
            muth[0]= 0 if aa== 0 else  muth1[ch][0]
            muth[1]= 0 if zz== 0 else  muth1[ch][1]
            muth[2]= 0 if ww== 0 else  muth1[ch][2]
            muth[3]= 0 if tata== 0 else  muth1[ch][3]
            muth[4]= 0 if mu[4]== 0 else  muth1[ch][4]
            errth[0]= 0 if aa== 0 else  errth1[ch][0]
            errth[1]= 0 if zz== 0 else errth1[ch][1]
            errth[2]= 0 if ww== 0 else  errth1[ch][2]
            errth[3]= 0 if tata== 0 else  errth1[ch][3]
            errth[4]= 0 if data['Bounds'][exp][ch]['mu_bb']== 0 else  errth1[ch][4]
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
    #    muExp=np.concatenate((muExp, [data['Bounds'][TopChannels][operator]]))
      #  errExp=np.concatenate((errExp, [data['Bounds'][TopChannels]['delta_'+operator]]))

    if TopChannels!=None:
        errTheo=np.concatenate((errTheo, [0.0]))

    
   
    num_of_obs=muExp.shape[0]



    #  correlations
    
    corr = np.identity(num_of_obs, dtype = float)
    #corrd = np.identity(num_of_obs, dtype = float)

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

    err=(errTheo**2+errExp**2)**0.5
    cov= err.T*corr*err
    # include correlations

        
        
        
    
    A = tt.dmatrix('A')
    A.tag.test_value = np.random.rand(2, 2)
    invA = tt.nlinalg.matrix_inverse(A)
    
    f = theano.function([theano.Param(A)], invA)   
    invcov=f(cov)
    #icv= invcov#- 1/(1+g)*np.matmul(invcov,np.matmul(B,invcov))
    iccov = 0.5*(invcov+invcov.T)
    deltamu = (muTheo-muExp).T
    chi2d=-0.5*np.matmul(np.matmul(deltamu,invcov),deltamu)
    return chi2d


