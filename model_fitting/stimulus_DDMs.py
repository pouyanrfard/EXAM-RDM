# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:06:05 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import re
import math
import random
import numpy as np
import pandas as pd
from numba import jit
from warnings import warn, filterwarnings
from rtmodels import rtmodel
import matplotlib.pyplot as plt

filterwarnings("always", message='This call to compute the log posterior', 
               category=ResourceWarning)

class leaky_accumulation_model(rtmodel):

    @property
    def D(self):
        """The dimensionality of the features space assumed by the model (read-only)."""
        return self._D
    
    @property
    def use_features(self):
        """Whether the model uses features as input, or just means."""
        if self._Trials.ndim == 1:
            return False
        else:
            return True
    
    @property
    def Trials(self):
        """Trial information used by the model.
        
        Either a 1D (use_features=False), or 3D (use_features=True) numpy array.
        When 1D, Trials contains the code of the correct choice in that trial.
        When 3D, Trials contains the stream of feature values that the subject
        may have seen in all the trials of the experiment. Then,
            S, D, L = Tials.shape
        where S is the length of the sequence in each trial
        """
        if self.use_features:
            return self._Trials
        else:
            return self.choices[self._Trials]
        
    @property
    def S(self):
        """Number of time steps maximally simulated by the model."""
        if self.Trials.ndim == 3:
            return self.Trials.shape[0]
        else:
            # the + 1 ensures that time outs can be generated
            return math.ceil(self.maxrt / self.dt) + 1
    
    @property
    def O(self):
        """Number of different conditions stored in the model."""
        return np.unique(self.conditions).size
    
    @property
    def conditions(self):
        """Indicates trial identities for differentiating parameters.
        
        Must be an array index consistent with the arrayed parameters in the 
        model.
        """
        return self._conditions
    
    def set_trials(self, Trials, conditions=0):
        """Setter for Trials and conditions."""
        Trials = np.array(Trials)
        conditions = np.array(conditions)
        
        if Trials.ndim == 3:
            self._Trials = Trials
            S, D, self._L = Trials.shape
            
            if self.D != D:
                warn('The dimensions of the input features in "Trials" ' + 
                     '(D=%d) do not match those stored in the model (D=%d)' % 
                     (D, self.D), RuntimeWarning)
        elif Trials.ndim == 1:
            # check that Trials only contains valid choice codes
            if np.all(np.in1d(np.unique(Trials), self.choices)):
                # transform to indices into self.choices
                self._Trials = np.array([np.flatnonzero(self.choices == i)[0] 
                                         for i in Trials])
                self._L = len(Trials)
            else:
                raise ValueError('Trials may only contain valid choice codes' +
                                 ' when features are not used.')
        else:
            raise ValueError('Trials has unknown format, please check!')
            
        if conditions.size == 1:
            self._conditions = np.full(self.L, conditions)
        elif conditions.size == self.L:
            self._conditions = conditions
        else:
            raise ValueError('conditions need to be scalar or array with one '
                             'element per trial!')
    
    @property
    def means(self):
        """Mean features values assumed in model."""
        return self._means
    
    @means.setter
    def means(self, means):
        if means.ndim == 1:
            means = means[None, :]
            
        D, C = means.shape
        
        if C != self.C:
            raise ValueError('The number of given means (%d) ' % C + 
                             'does not match the number of choices (%d) ' % self.C +
                             'processed by the model')
        else:
            self._means = means
            if self._D != D and self.use_features:
                warn('The given means changed the dimensionality of the ' + 
                     'model. Update Trials!')
            self._D = D

    parnames = ['scale', 'scalestd', 'bound', 'noisestd', 'bias', 'biasstd','ndtmean', 'ndtspread', 
                'leak', 'lapseprob','lapsetoprob']

    @property
    def P(self):
        "number of parameters in the model"
        
        # the prior adds C-1 parameters, one of which is counted by its name
        return len(self.parnames)
    
    @property
    def condpars(self):
        """Names of the parameters that vary with condition."""
        names = []
        for name in self.parnames:
            if np.array(getattr(self, name)).size > 1:
                names.append(name)
                
        return names

    indexedpar_re = re.compile(r'([a-zA-Z_]*[a-zA-Z]+)_?(\d)?')

    def __init__(self, Trials, conditions=0, dt=0.1, means=None, scale=1, 
                 scalestd=0, bound=0.1, bias=0, biasstd=0, noisestd=0.1, ndtmean=0, 
                 leak=0, ndtspread=0, lapseprob=0.05, lapsetoprob=0.1, 
                 **rtmodel_args):
        super(leaky_accumulation_model, self).__init__(**rtmodel_args)
        
        self.name = 'Leaky accumulation model'
        
        # Time resolution of model simulations.
        self.dt = dt
        
        # Trial information used by the model.
        Trials = np.array(Trials)
        
        # try to figure out the dimensionality of feature space from means
        D_mean = None
        if means is not None:
            means = np.array(means)
            if means.ndim == 1:
                D_mean = 1
            else:
                D_mean = means.shape[0]
        
        # check with dimensionality of feature space from Trials
        if Trials.ndim == 1:
            if D_mean is None:
                # no information provided by user
                self._D = 1
            else:
                self._D = D_mean
        elif Trials.ndim == 3:
            if D_mean is None: 
                self._D = Trials.shape[1]
            else:
                if Trials.shape[1] == D_mean:
                    self._D = Trials.shape[1]
                else:
                    raise ValueError("The dimensionality of the provided "
                                     "means and features in Trials is not "
                                     "consistent!")
        
        # now set Trials internally (because dimensionality of feature space is
        # set before, there should be no warnings)
        self.set_trials(Trials, conditions)
        
        if means is not None:
            self.means = means
        elif self.D == 1:
            self._means = np.arange(self.C)
            self._means = self._means - np.mean(self._means)
            self._means = self._means[None, :]
        elif self.D == 2:
            phi = math.pi/4 + np.linspace(0, 2*math.pi * (1-1/self.C), self.C)
            self._means = np.r_[np.cos(phi), np.sin(phi)];
        else:
            warn('Cannot set default means. Please provide means yourself!',
                 RuntimeWarning)
                 
        # scale of feature values during accumulation
        self.scale = scale
        
        # standard deviation of feature values during accumulation
        self.scalestd = scalestd
        
        # Bound that needs to be reached before decision is made.
        self.bound = bound
        
        # Standard deviation of noise added to feature values.
        self.noisestd = noisestd
            
        # leak parameter which is the discount rate of evidence accumulation
        self.leak = leak
        
        # bias toward one of the choices.
        self.bias = bias
        
        # tandard deviation of bias
        self.biasstd = biasstd
            
        # Mean of nondecision time.
        self.ndtmean = ndtmean
            
        # Spread of nondecision time.
        self.ndtspread = ndtspread
            
        # Probability of a lapse.
        self.lapseprob = lapseprob
            
        # Probability that a lapse will be timed out.
        self.lapsetoprob = lapsetoprob
            
    
    def estimate_memory_for_gen_response(self, N):
        """Estimate how much memory you would need to produce the desired responses."""
        
        mbpernum = 8 / 1024 / 1024
        
        # (for input features + for input params + for output responses)
        return mbpernum * N * (self.D * self.S + self.P + 2)
    
    
    def gen_response(self, trind, rep=1):
        N = trind.size
        if rep > 1:
            trind = np.tile(trind, rep)
        
        choices, rts = self.gen_response_with_params(trind)
        
        if rep > 1:
            choices = choices.reshape((N, rep), order='F')
            rts = rts.reshape((N, rep), order='F')
        
        return choices, rts
        
        
    def gen_response_with_params(self, trind, params={}, parnames=None, 
                                 user_code=True):
        condpars = self.condpars
        
        if parnames is None:
            assert isinstance(params, dict)
            pardict = params
        else:
            assert isinstance(params, np.ndarray)
            
            isnew = {name: True for name in condpars}
            
            pardict = {}
            for ind, name in enumerate(parnames):
                name, pind = self.indexedpar_re.match(name).groups()
                
                if pind is None:
                    pardict[name] = params[:, ind]
                else:
                    if isnew[name]:
                        pardict[name] = np.tile(getattr(self, name), 
                                                (params.shape[0], 1))
                        isnew[name] = False
                        
                    pardict[name][:, int(pind)] = params[:, ind]
                    
        parnames = pardict.keys()
        
        # get the number of different parameter sets, P, and check whether the 
        # given parameter counts are consistent (all have the same P)
        P = None
        for name in parnames:
            if not np.isscalar(pardict[name]):
                if P is None:
                    P = pardict[name].shape[0]
                else:
                    if P != pardict[name].shape[0]:
                        raise ValueError('The given parameter dictionary ' +
                            'contains inconsistent parameter counts')
        if P is None:
            P = 1
        
        # get the number of trials, N, and check whether it is consistent with 
        # the number of parameters P
        if np.isscalar(trind):
            trind = np.full(P, trind, dtype=int)
        N = trind.shape[0]
        if P > 1 and N > 1 and N != P:
            raise ValueError('The number of trials in trind and the ' +
                             'number of parameters in params does not ' + 
                             'fit together')
        
        NP = max(N, P)
        
        # if continuing would exceed the memory limit
        if self.estimate_memory_for_gen_response(NP) > self.memlim:
            # divide the job in smaller batches and run those
        
            # determine batch size for given memory limit
            NB = math.floor(NP / self.estimate_memory_for_gen_response(NP) *
                            self.memlim)
            
            choices = np.zeros(NP, dtype=np.int8)
            rts = np.zeros(NP)
            
            remaining = NP
            firstind = 0
            while remaining > 0:
                index = np.arange(firstind, firstind + min(remaining, NB))
                if P > 1 and N > 1:
                    trind_batch = trind[index]
                    params_batch = extract_param_batch(pardict, index)
                elif N == 1:
                    trind_batch = trind
                    params_batch = extract_param_batch(pardict, index)
                elif P == 1:
                    trind_batch = trind[index]
                    params_batch = pardict
                else:
                    raise RuntimeError("N and P are not consistent.")
                    
                choices[index], rts[index] = self.gen_response_with_params(
                    trind_batch, params_batch, user_code=user_code)
                
                remaining -= NB
                firstind += NB
        else:
            # make a complete parameter dictionary with all parameters
            # this is quite a bit of waste of memory and should probably be recoded
            # more sensibly in the future, but for now it makes the jitted function
            # simple
            allpars = {}
            for name in self.parnames:
                if name in parnames:
                    allpars[name] = pardict[name]
                else:
                    # get parameter value stored in model and repeat P-times
                    if name in condpars:
                        allpars[name] = np.tile(getattr(self, name), (P, 1))
                    else:
                        allpars[name] = np.tile(getattr(self, name), P)
                    
                # expand parameter values N-times, if only one set given
                if P == 1 and N > 1:
                    if name in condpars:
                        allpars[name] = np.tile(allpars[name], (N, 1))
                    else:
                        allpars[name] = np.tile(allpars[name], N)
                
                # select the condition-specific parameter values
                if name in condpars:
                    allpars[name] = allpars[name][
                            np.arange(NP), self.conditions[trind]]
            
            # select input features
            if self.use_features:
                features = self.Trials[:, :, trind]
            else:
                features = self.means[:, self._Trials[trind]]
                features = np.tile(features, (self.S, 1, 1))
                
            # call the compiled function
            choices, rts = self.gen_response_jitted(features, allpars)
                
            # transform choices to those expected by user, if necessary
            if user_code:
                toresponse_intern = np.r_[-1, self.toresponse[1]]
                timed_out = choices == toresponse_intern[0]
                choices[timed_out] = self.toresponse[0]
                in_time = np.logical_not(timed_out)
                choices[in_time] = self.choices[choices[in_time]]
            
        return choices, rts
        
        
    def gen_response_jitted(self, features, allpars):
        toresponse_intern = np.r_[-1, self.toresponse[1]]
            
        # call the compiled function
        choicesOut, rts = gen_response_jitted_lam(
                self.dt, features, self.S, toresponse_intern, self.maxrt, 
                allpars['scale'],allpars['scalestd'], allpars['bound'], allpars['bias'], 
                allpars['biasstd'],allpars['noisestd'], allpars['ndtmean'], allpars['ndtspread'], 
                allpars['leak'], allpars['lapseprob'],allpars['lapsetoprob'])
        
        return choicesOut, rts
        
    
    def plot_response_distribution(self, choice, rt, sepcond=False, ax=None, 
                                   **kwargs):
        if sepcond:
            if ax is None:
                fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            
            assert isinstance(ax, (list, np.ndarray)) and len(ax) == self.O
            assert choice.shape[0] == self.L
            
            if choice.ndim == 1:
                choice = choice[:, None]
                rt = rt[:, None]
            
            for o, a in zip(range(self.O), ax):
                super(leaky_accumulation_model, self).plot_response_distribution(
                        choice[self.conditions == o, :], 
                        rt[self.conditions == o, :], ax=a, **kwargs)
                
                a.set_title('condition %d' % o)
        else:
            ax = super(leaky_accumulation_model, 
                       self).plot_response_distribution(
                               choice, rt, ax=ax, **kwargs)
            
        return ax
    
    
    def plot_parameter_distribution(self, samples, names, q_lower=0, q_upper=1,
                                    distribution='prior'):
        samples = samples.copy()
        
        if type(samples) is not pd.DataFrame:
            samples = pd.DataFrame(samples, columns=names)
            
        if 'distribution' not in samples.columns:
            samples['distribution'] = distribution
        
        # when non-decision time distribution is inferred,
        # transform the parameter samples to samples of the mode and std of 
        # that distribution
        if 'ndtmean' in samples.columns and 'ndtspread' in samples.columns:
            samples['ndtmode'] = np.exp(samples['ndtmean'] - samples['ndtspread']**2)
            samples['ndtstd'] = np.sqrt( (np.exp(samples['ndtspread']**2) - 1) * 
                np.exp(2 * samples['ndtmean'] + samples['ndtspread']**2) )
            samples.drop(['ndtmean', 'ndtspread'], axis=1, inplace=True)
            
        super(leaky_accumulation_model, self).plot_parameter_distribution(
            samples, samples.columns.drop('distribution'), q_lower, q_upper)


    def __str__(self):
        info = super(leaky_accumulation_model, self).__str__()
        
        # empty line
        info += '\n'
        
        # model-specific parameters
        info += 'means:\n'
        info += self.means.__str__() + '\n'
        info += ' uses features: %8d' % self.use_features + '\n'
        info += '            dt: %8.3f' % self.dt + '\n'
        
        for name in self.parnames:
            if name in self.condpars:
                info += '{:>14}: {: 8.3f} (0)\n'.format(
                        name, getattr(self, name)[0])
                for cond in range(1, self.O):
                    info += '{:>14}  {: 8.3f} ({:d})\n'.format(
                        '', getattr(self, name)[cond], cond)
            else:
                info += '{:>14}: {: 8.3f}\n'.format(name, getattr(self, name))
        
        return info


@jit(nopython=True, cache=True)
def gen_response_jitted_lam(dt, features, S, toresponse, maxrt, scale, scalestd,
                            bound, bias, biasstd, noisestd, ndtmean, ndtspread, leak,
                            lapseprob, lapsetoprob):
    
    S, D, N = features.shape
    if D > 1:
        raise NotImplementedError("Currently the leaky accumulation model "
                                  "is only implemented for 1D features.")
    C=2
    
    # initialise responses with time-out responses
    choicesOut = np.full(N, toresponse[0], dtype=np.int32)
    rts = np.full(N, toresponse[1])
    
    # go through all trials
    for tr in range(N):
        
        scaleTr = random.normalvariate(scale[tr], scalestd[tr])
        scaleTr = max(scaleTr, 0)
        
        # is it a lapse trial?
        if random.random() < lapseprob[tr]:
            # if it is not a timed-out lapse trial
            if random.random() >= lapsetoprob[tr]:
                # overwrite the time-out response with a random response
                choicesOut[tr] = random.randint(0, C-1)
                rts[tr] = random.random() * maxrt
        else:
            
            biasTrVal=np.random.uniform(bias[tr] - biasstd[tr] / 2,
                                         bias[tr] + biasstd[tr] / 2)
            
                
            logEVTr = biasTrVal * bound[tr]
                    
            for t in range(1, S+1):
                # if a current feature is -Inf, respond with time out (default)
                # this is the case when the participant responded before the
                # present time point had been reached
                if features[t-1, 0, tr] == -np.inf:
                    break
                
                
                # add normal noise to the present feature value
                noisyFeature = (  features[t-1, 0, tr] * dt * scaleTr
                                + random.normalvariate(0, noisestd[tr]) 
                                * np.sqrt(dt))
                
                # accumulate with leak
                logEVTr = (1 - leak[tr]) * logEVTr + noisyFeature
                
                # if evidence reaches upper or lower bound, record response
                if abs(logEVTr) > bound[tr]:
                    # without non-decision time the RT is the time point just 
                    # after the time bin in which the present feature occurred,
                    # so it's simply the number of already seen features * dt
                    NdtTr = np.random.uniform(ndtmean[tr] - ndtspread[tr] / 2, 
                          ndtmean[tr] + ndtspread[tr] / 2)
                    
                    rts[tr] = t * dt + NdtTr
                    
                    # choices should be 0 and 1 which will be mapped to the
                    # user-defined choices afterwards (if self.choices=[-1, 1],
                    # 0 will be mapped to -1 and 1 to 1, i.e., the choices here
                    # are indeces into the user-defined choice vector)
                    if logEVTr >= 0:
                        choicesOut[tr] = 1
                    else:
                        choicesOut[tr] = 0
                    break
                
            if rts[tr] > maxrt:
                choicesOut[tr] = toresponse[0]
                rts[tr] = toresponse[1]
            
    return choicesOut, rts


def extract_param_batch(pardict, index):
    newdict = {}
    for parname, values in pardict.items():
        if values.ndim == 2:
            newdict[parname] = values[index, :]
        else:
            newdict[parname] = values[index]
            
    return newdict