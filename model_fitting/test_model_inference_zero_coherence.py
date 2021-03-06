#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:43:46 2018

@author: Sebastian Bitzer
@authoer: Pouyan Rafieifard
"""

import numpy as np
import pyEPABC
from pyEPABC.parameters import exponential, gaussprob, zero
import stimulus_DDMs
import scipy.io as sio
import matplotlib.pyplot as plt



def run_behavioral_fit(sub,cond,aper,featureType,run,recovery, 
                       diagnostic_plots=False):

#%% load some data
    if(featureType==0):
        loadFileName="..\\test_model_fit_data\\fitData_sub_%d_cond_%d_me_norm_final_2.mat" % (sub,cond)
    elif(featureType==1):
        loadFileName="..\\test_model_fit_data\\fitData_sub_%d_cond_%d_dc_norm_final_2.mat" % (sub,cond)
        
    mat_contents=sio.loadmat(loadFileName,squeeze_me=True)
    
    N = 240
    
    dt = 1/60
    maxrt = 2
    
    S = int(maxrt / dt)
    modelchoices = [-1, 1]
    
    means = np.r_[-0.1, 0.1]
    
    choices=mat_contents['choices']
    rts=mat_contents['rts']
    conditions=mat_contents['conditions']
    
    
    if(aper==0):
        features=mat_contents['averageFeatures']
    else:
        features=mat_contents['features']
    
    
    featuresIn=np.empty((N,S+1))
    featBaseline=-np.inf
    
    for tr in range(N):
        for t in range(S+1):
            if(t<len(features[tr])):
                featuresIn[tr][t]=features[tr][t]
            else:
                featuresIn[tr][t]=featBaseline

    
           
    featuresIn=np.transpose(featuresIn)
    featuresOut=np.reshape(featuresIn,(featuresIn.shape[0],1,featuresIn.shape[1]))
    
    Trials=featuresOut
        
    #%% fit the data with pyEPABC
    # make a fit model so that you can set it with independent parameters
    fitmodel = stimulus_DDMs.leaky_accumulation_model(
            Trials, conditions, dt, means, maxrt=maxrt, choices=[-1, 1], 
            toresponse=[0, 5.0], noisestd=0.1, scale=np.zeros(2))
    
    pars = pyEPABC.parameters.parameter_container()
    # add scale twice with the same transform and prior adding scale_0 and scale_1
    
    pars.add_param('scale', 0, 0.1, zero(), multiples=2)#0,0.1
    pars.add_param('scalestd', np.log(.05), 0.5, exponential())
    pars.add_param('bound', np.log(.05), 1, exponential())
    pars.add_param('bias', 0, 0.5)
    pars.add_param('biasstd', np.log(0.02), 0.5,exponential())
    pars.add_param('ndtmean', -2, 1,exponential())
    pars.add_param('ndtspread', -2.5, 1, exponential())
    #pars.add_param('leak', -1, 1, gaussprob())
    pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
    pars.add_param('lapsetoprob', 0, 1, gaussprob())
    
    pg = pars.plot_param_dist();
    pg.fig.tight_layout()
    
    
    simfun = lambda data, dind, parsamples: fitmodel.gen_distances_with_params(
                data[0], data[1], dind, pars.transform(parsamples), pars.names)
        
    epsilon = 0.05
    veps = 2 * epsilon
    
    # calling EPABC:
    ds = 1
    ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = pyEPABC.run_EPABC(
            np.c_[choices, rts], simfun, None, pars.mu, pars.cov, 
            epsilon=epsilon, minacc=2000, samplestep=10000, samplemax=6000000, 
            npass=2, alpha=0.5, veps=veps)
    
    
    #%% analysis
    # pairwise parameter posteriors
    pg = pars.plot_param_dist(ep_mean, ep_cov)
    pg.fig.tight_layout()
    
    if(aper==0):
        picName = "..\\test_model_fit_data\\fitPosterior_sub_%d_cond_%d_aper_%d_run_%d_rw_final_16_3.png" % (sub,cond,aper,run)
    else:
        if(featureType==0):
            picName = "..\\test_model_fit_data\\fitPosterior_sub_%d_cond_%d_aper_%d_run_%d_me_final_16_3.png" % (sub,cond,aper,run)
        elif(featureType==1):
            picName = "..\\test_model_fit_data\\fitPosterior_sub_%d_cond_%d_aper_%d_run_%d_dc_final_16_3.png" % (sub,cond,aper,run)
      
  
    plt.savefig(picName,bbox_inches='tight')
    
    # compare the posterior pdfs of the two scales
    if 'scale_0' in pars.names.values and 'scale_1' in pars.names.values:
        fig, ax = plt.subplots()
        scale_0 = pars.params[pars.params.name == 'scale_0']['transform']
        ind_0 = scale_0.index[0]
        scale_0 = scale_0.loc[ind_0]
        scale_1 = pars.params[pars.params.name == 'scale_1']['transform']
        ind_1 = scale_1.index[0]
        scale_1 = scale_1.loc[ind_1]
        
        qs0 = scale_0.transformed_ppf(
                [0.025, 0.99], ep_mean[ind_0], ep_cov[ind_0, ind_0])
        qs1 = scale_1.transformed_ppf(
                [0.025, 0.99], ep_mean[ind_1], ep_cov[ind_1, ind_1])
        
        xx = np.linspace(min(qs0[0], qs1[0]), max(qs0[1], qs1[1]), 2000)
        
        ax.plot(xx, scale_0.transformed_pdf(
                xx, ep_mean[ind_0], ep_cov[ind_0, ind_0]), label='scale_0')
        ax.plot(xx, scale_1.transformed_pdf(
                xx, ep_mean[ind_1], ep_cov[ind_1, ind_1]), label='scale_1')
        #ax.plot(model.scale[0], 0, '*', color='C0', label='true')
        #ax.plot(model.scale[1], 0, '*', color='C1')
        ax.legend()
        
        p_0 = 1 - scale_0.transformed_pdf(0, ep_mean[ind_0], ep_cov[ind_0, ind_0])
        p_1 = 1 - scale_1.transformed_pdf(0, ep_mean[ind_1], ep_cov[ind_1, ind_1])
        print('posterior probability that scale > 0: %5.3f (0) vs %5.3f (1)'
              % (p_0, p_1))
        
        
        if(featureType==0):
            picName = "..\\test_model_fit_data\\fitScalePosterior_sub_%d_cond_%d_aper_%d_run_%d_me_cond_final_16_3.png" % (sub,cond,aper,run)
        elif(featureType==1):
            picName = "..\\test_model_fit_data\\fitScalePosterior_sub_%d_cond_%d_aper_%d_run_%d_dc_cond_final_16_3.png" % (sub,cond,aper,run)
                
        
        plt.savefig(picName, bbox_inches='tight')
        
    
       
    if(aper>0):
        if(featureType==0):
            saveFileName="..\\test_model_fit_data\\fitResults_sub_%d_cond_%d_aper_%d_run_%d_me_cond_final_16_3.mat" % (sub,cond,aper,run)
        elif(featureType==1):
            saveFileName="..\\test_model_fit_data\\fitResults_sub_%d_cond_%d_aper_%d_run_%d_dc_cond_final_16_3.mat" % (sub,cond,aper,run)
    else:
        saveFileName="..\\test_model_fit_data\\fitResults_sub_%d_cond_%d_aper_%d_run_%d_rw_cond_final_16_3.mat" % (sub,cond,aper,run)
            
    sio.savemat(saveFileName,mdict={'sub':sub, 'cond':cond,'ep_mean':ep_mean, 'ep_cov':ep_cov, 'ep_logml':ep_logml,
                                    'nacc':nacc,'ntotal':ntotal,'p_0':p_0,'p_1':p_1})                                  
    
    
    

if __name__ == "__main__":

        #selected participants according to the criteria mentioned in the Methods (see manuscript)    
        selSubs=np.array([62,63,64,65,66,67,71,74,76,77,80,82,84,85,86,90,91,92,93,94,95,96,
                          97,100,101,102,104,105,107,109,110,111,112,114,119,120,121,123,124,126,127,128,129,130])
        
        
        
        #specifiying the parameters of model fitting
        conditions=np.array([1]) #1 for 0% coherence condition
        apers=np.array([0,12]) #0 for DDM, 12 for EXaM
        featuresTypes=np.array([0,1]) #0 for motion energy, 1 for dot counts
        recovery=0
                
        for subIdx in range(len(selSubs)): #subjects
            #reset the counter for model runs in every iteration
            run=1
            while run<=5:        
                try:
                    #if the model fitting runs without exceptions go to next iteration
                    sub=selSubs[subIdx]
                    cond=conditions[0]                    
                    aper=apers[1]
                    featureType=featuresTypes[1]
                    print('running fit for sub: ',sub,' cond: ',cond,' aper: ',aper,'feature:',featureType, ' run:', run)                    
                    run_behavioral_fit(sub,cond,aper,featureType,run,recovery)
                    run=run+1
                except:
                    print('\n-------------------------')
                    print('EP-ABC model fitting interrupted, re-runing the model fit')
                    print('-------------------------')
                    pass
                        

