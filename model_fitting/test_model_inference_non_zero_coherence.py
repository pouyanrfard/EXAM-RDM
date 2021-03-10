#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:48:12 2017

@author: ُSebastian Bitzer
@author: Pouyan Rafieifard
"""

import numpy as np
import pandas as pd
import pyEPABC
from pyEPABC.parameters import exponential, gaussprob, zero
import stimulus_DDMs
import scipy.io as sio
import matplotlib.pyplot as plt

import seaborn as sns


#%% sample from the posterior ndt distribution
def get_ndtsamples(pars, mu=None, cov=None):
    samples = pd.DataFrame(pars.sample_transformed(2000, mu, cov), 
                           columns=pars.names)
    samples = samples.drop(
            np.setdiff1d(samples.columns, ['ndtmean', 'ndtspread']), 
            axis=1)
    
    return samples.apply(
            lambda row: (np.random.rand() - 0.5) * row.ndtspread + row.ndtmean, 
            axis=1)
    
    
def plot_singletrial_fit(ax, ch_sim, rt_sim, ch, rt, model):
    model.plot_response_distribution(ch_sim, rt_sim, ax=ax)
    
    chind = np.flatnonzero(model.choices == ch)
    if chind.size > 0:
        cols = ['C0', 'C1']
        col = cols[chind[0]]
    else:
        col = 'k'
    
    yl = ax.get_ylim()
    ax.plot(rt, -0.05*yl[1], '*', color=col)
    ax.set_ylim(-0.1*yl[1], yl[1])
    

def plot_singletrials_highres(trials, choices, rts, model, pars, mu, cov, 
                              res=20000):
    ch_sim, rt_sim = model.gen_response_from_Gauss_posterior(
            trials, pars.names, mu, cov, res, pars.transform)
    
    fig, axes = plt.subplots(1, trials.size, sharex=True, sharey=True)
    
    for ax, ind in zip(axes, range(trials.size)):
        plot_singletrial_fit(ax, ch_sim[ind, :], rt_sim[ind, :], 
                             choices[trials[ind]], rts[trials[ind]], model)
        
        ax.set_xlabel('RT')
        ax.set_title('trial %d' % trials[ind])
        
    axes[0].set_ylabel('density value')
    
    return fig, axes


#%% load trials

def run_behavioral_fit(sub,cond,aper,featureType,run,recovery, 
                       diagnostic_plots=False):
    
    #featureType=0 motion energy, featureType=1 for dot counts
    
    if(featureType==0):
        loadFileName="..//test_model_fit_data//fitData_sub_%d_cond_%d_me_norm_final_2.mat" % (sub,cond)
    elif(featureType==1):
        loadFileName="..//test_model_fit_data//fitData_sub_%d_cond_%d_dc_norm_final_2.mat" % (sub,cond)
        
    mat_contents=sio.loadmat(loadFileName,squeeze_me=True)
    
    #%% create a test model and simulate responses
    if(cond<3):
        N = 240
    else:
        N=160
    
    
    dt = 1/60
    maxrt = 2
    
    S = int(maxrt / dt)
    modelchoices = [-1, 1]
    
    means = np.r_[-0.1, 0.1]

    
    
    features=mat_contents['features']
    
    
    featuresIn=np.empty((N,S))
    featBaseline=-np.inf
    
    if(aper>0):
        for tr in range(N):
            for t in range(S):
                if(t<len(features[tr])):
                    featuresIn[tr][t]=features[tr][t]
                else:
                    featuresIn[tr][t]=featBaseline
    else:
        trialMeans = means[np.random.randint(2, size=N)]
        for tr in range(N):
           featuresIn[tr,:]=trialMeans[tr]*np.ones(S)
    
           
    featuresIn=np.transpose(featuresIn)
    featuresOut=np.reshape(featuresIn,(featuresIn.shape[0],1,featuresIn.shape[1]))
    
    #print(featuresOut)
    
    if(aper>0):
        Trials=featuresOut
    else:
        Trials = featuresOut / np.std(featuresOut)
    
    #%% create model
    
    model = stimulus_DDMs.leaky_accumulation_model(
            Trials, dt, means, scale=0.01, ndtmean=0.6, ndtspread=0.2, 
            noisestd=0.1, bound=.1, leak=0, bias=0.02, scalestd=0,
            choices=modelchoices, maxrt=maxrt, toresponse=[0, 5.0])
    
    
    if(recovery==0):
        choices=mat_contents['choices']
        rts=mat_contents['rts']
        
        ax = model.plot_response_distribution(choices, rts)
    else:
        # generate simulated responses
        fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
        for row in axes:
            for ax in row:
                choices, rts = model.gen_response(np.arange(N))
                #model.plot_response_distribution(choices, rts, ax=ax)
    
    
    #%% make a fit model and provide features that are clipped after the response
    times = np.tile((np.arange(1, S+1) * dt)[:, None, None], (1, 1, N))
    Trials_fit = Trials.copy()
#    np.random.shuffle(Trials_fit.T)
    Trials_fit[times > rts] = -np.inf
    
    fitmodel = stimulus_DDMs.leaky_accumulation_model(
            Trials_fit, 0, dt, means, choices=modelchoices, maxrt=maxrt, 
            leak=0, scalestd=0, noisestd=0.1, toresponse=[0, 5.0])
    
    
    #%% setup parameters
    pars = pyEPABC.parameters.parameter_container()
    if aper > 0:
        pars.add_param('scale', 0, 1, zero())
    else:
        fitmodel.scale = 0
#    pars.add_param('scale', 0, 1, gaussprob())
    #pars.add_param('scale', 0.5, 0.1)
#    pars.add_param('scalestd', np.log(.1), 1, exponential())
    pars.add_param('bound', np.log(.1), 1, exponential())
    #pars.add_param('noisestd', np.log(.1), 1, exponential())
    pars.add_param('bias', 0, 0.5)
    pars.add_param('ndtmean', -2, 1,exponential())
    pars.add_param('ndtspread', -2.5, 1, exponential())
    #pars.add_param('leak', -1, 1, gaussprob())
    pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
    pars.add_param('lapsetoprob', 0, 1, gaussprob())
    
    pg = pars.plot_param_dist();
    pg.fig.tight_layout()
    
    if(sub==62):
        if(aper==0):
            plt.savefig('examplePrior_rw.png', bbox_inches='tight')
    
        if(aper==12):
            plt.savefig('examplePrior_dc.png', bbox_inches='tight')
    
    #%% fit the model to the simulated data
    # simulation function:
    simfun = lambda data, dind, parsamples: fitmodel.gen_distances_with_params(
            data[0], data[1], dind, pars.transform(parsamples), pars.names)
    
    epsilon = 0.05
    veps = 2 * epsilon
    
    # calling EPABC:
    logmls = np.zeros(10)
    for i in range(1):
        ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = pyEPABC.run_EPABC(
                np.c_[choices, rts], simfun, None, pars.mu, pars.cov, epsilon=epsilon, 
                minacc=2000, samplestep=10000, samplemax=6000000, npass=2, 
                alpha=0.5, veps=veps)
        logmls[i] = ep_logml
    
    #%% checking posterior:
    pg = pars.plot_param_dist(ep_mean, ep_cov)
    pg.fig.tight_layout()
    
    if diagnostic_plots:
        fig, axes = pars.compare_pdfs(ep_mean, ep_cov, figsize=[10, 6])
        if 'scale' in pars.names.values:
            scaleind = pars.names[pars.names == 'scale'].index[0]
            print('p(scale > 0) = %5.3f' % (1 - zero().transformed_pdf(
                    0, ep_mean[scaleind], ep_cov[scaleind, scaleind])))
    
    #%% save
    if(aper==0):
        picName = "..//test_model_fit_data//fitPosterior_sub_%d_cond_%d_aper_%d_run_%d_rw_final_16.png" % (sub,cond,aper,run)
    else:
        if(featureType==0):
            picName = "..//test_model_fit_data//fitPosterior_sub_%d_cond_%d_aper_%d_run_%d_me_final_16.png" % (sub,cond,aper,run)
        elif(featureType==1):
            picName = "..//test_model_fit_data//fitPosterior_sub_%d_cond_%d_aper_%d_run_%d_dc_final_16.png" % (sub,cond,aper,run)
      
    plt.savefig(picName, bbox_inches='tight')

    
    # plot true parameter values for comparison
#    for pname, ax in zip(pars.names, pg.diag_axes):
#        ax.plot(np.ones(2)*model.__dict__[pname], [0, 0.1 * ax.get_ylim()[1]], 'r')
#        ax.set_ylim(bottom=0)
    
    #%% plot fitted RT distribution
    if diagnostic_plots:
        fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(4, 9))
        choices_post, rts_post = fitmodel.gen_response_from_Gauss_posterior(
                np.arange(N), pars.names, ep_mean, ep_cov, 500, pars.transform)
        fitmodel.plot_response_distribution(choices, rts, ax=axes[0])
        axes[0].set_ylabel('subject responses')
        fitmodel.plot_response_distribution(choices_post, rts_post, ax=axes[1])
        axes[1].set_ylabel('posterior predictive')
        mu_dec = ep_mean.copy()
        ndind = pd.Index(pars.params.name).get_indexer(['ndtmean', 'ndtspread', 'lapseprob'])
        mu_dec[ndind] = -100
        cov_dec = ep_cov.copy()
        cov_dec[ndind, ndind] = 1e-10
        choices_dec, rts_dec = fitmodel.gen_response_from_Gauss_posterior(
                np.arange(N), pars.names, mu_dec, cov_dec, 100, pars.transform)
        fitmodel.plot_response_distribution(choices_dec, rts_dec, ax=axes[2])
        axes[2].set_ylabel('posterior dts')
        sns.distplot(get_ndtsamples(pars, ep_mean, ep_cov), kde=False, 
                     norm_hist=True, ax=axes[3])
        axes[3].set_ylabel('posterior ndts')
        axes[0].set_title('across all trials')
        fig.tight_layout()
    
    #%% plot the posterior predictive distribution for some trials
    if diagnostic_plots:
        plottr = np.random.randint(N, size=4)
        fig, axes = plot_singletrials_highres(plottr, choices, rts, fitmodel, 
                                              pars, ep_mean, ep_cov)
        fig.set_size_inches(10, 3.5)
        fig.tight_layout()
        
    
    #%% compare predictive log-likelihoods 
    if diagnostic_plots:
        logpls, _ = pyEPABC.estimate_predlik(
                np.c_[choices, rts], simfun, ep_mean, ep_cov, epsilon)
    
        if 'scale' in pars.names.values:
            # set scale to very small, i.e., transformed scale ~= 0
            ep_mean0 = ep_mean.copy()
            scaleind = pars.names[pars.names == 'scale'].index[0]
            ep_mean0[scaleind] = -1000
            
            logpls0, _ = pyEPABC.estimate_predlik(
                    np.c_[choices, rts], simfun, ep_mean0, ep_cov, epsilon)
            
            print('predictive log-lik sum: % 7.2f vs % 7.2f (posterior scale '
                  'vs scale = 0)' % (logpls.sum(), logpls0.sum()))
        else:
            print('predictive log-lik sum: % 7.2f' % logpls.sum())
        
    
    #%% getting modes of transformed parameter posterior:
    modes = pars.get_transformed_mode(ep_mean, ep_cov)
    
    if(aper>0):
        if(featureType==0):
            saveFileName="..//test_model_fit_data//fitResults_sub_%d_cond_%d_aper_%d_run_%d_me_final_16.mat" % (sub,cond,aper,run)
        elif(featureType==1):
            saveFileName="..//test_model_fit_data//fitResults_sub_%d_cond_%d_aper_%d_run_%d_dc_final_16.mat" % (sub,cond,aper,run)
    else:
        saveFileName="..//test_model_fit_data//fitResults_sub_%d_cond_%d_aper_%d_run_%d_rw_final_16.mat" % (sub,cond,aper,run)
        
    sio.savemat(saveFileName,mdict={'sub':sub, 'cond':cond,'ep_mean':ep_mean, 'ep_cov':ep_cov, 'ep_logml':ep_logml,
                                    'nacc':nacc,'ntotal':ntotal,'modes':modes})                                  
    
    #picName = "fitData//fitPosterior_sub_%d_cond_%d_aper_%d_new_2.png" % (sub,cond,aper)
    
    

if __name__ == "__main__":
        

        selSubs=np.array([62,63,64,65,66,67,71,74,76,77,80,82,84,85,86,90,91,92,93,94,95,96,
                  97,100,101,102,104,105,107,109,110,111,112,114])
        conditions=np.array([1,2,3,4])
        apers=np.array([0,12]) #0,3,5,12
        recovery=0
        
        #run_behavioral_fit(15,1,0)  
        featuresTypes=np.array([1]) #0 for motion energy, 1 for dot counts
        
        for subIdx in range(10,11):#len(selSubs)): #subjects
            for condIdx in range(len(conditions)): #conditions
                for runIdx in range(5,10):
                    
                    run=runIdx+1
                                        
                    #fit the random walk model only once to the data
                    sub=selSubs[subIdx]
                    cond=conditions[condIdx]                    
                    aper=apers[0]
                    featureType=featuresTypes[0]
                    print('running fit for sub: ',sub,' cond: ',cond,' aper: ',aper,'feature',featureType)                    
                    run_behavioral_fit(sub,cond,aper,featureType,run,recovery)  

