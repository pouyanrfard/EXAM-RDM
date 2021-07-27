'''
The script 'plot_prior_and_posterior_examples.py' creates the visualizations related 
the modeling priors and psoteriors for Figure 9 in "Fard et al. (2021), Spatiotemporal 
% Modeling of Response Consistency with Incoherent Motion Stimuli in 
% Perceptual Decision Making, In submission".

%Hint: Please make sure that this script is placed next to the folder
%"behavioral_data", "analysis_data", "experiment_design", and "functions".
%The script may take a few minutes to run depending on your system
%specifications. Also, please check the "Figures" folder for the outcomes of 
the visualization.

Author: Pouyan Rafieifard (January 2021)
'''

#################################
#loading the necessary python libraries
#################################

#%matplotlib inline
from pylab import *
import pylab
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.io as sio
from matplotlib.ticker import AutoMinorLocator
sns.set(style="white", color_codes=True,font_scale=0.7)
from scipy.stats import lognorm
from scipy.stats import uniform
from transforms import gaussprobpdf
from scipy.stats import norm

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'


#################################
#Calculating the prior distributions
#################################

prior_mean = np.array([0, 5, -10, 0, -2, -1.5, -2, -1, 0])
prior_cov = np.diag(np.array([1, 1, 3, 1, 1, 1, 1, 1, 1]) ** 2)


paramtransform = lambda params: np.c_[norm.cdf(params[:, 0]) / 2 + 0.5,
                                      np.exp(params[:, 1]),
                                      np.exp(params[:, 2]),
                                      norm.cdf(params[:, 3]),
                                      norm.cdf(params[:, 4]),
                                      np.exp(params[:, 5]),
                                      np.exp(params[:, 6]),
                                      norm.cdf(params[:, 7:])]


#################################
#Creating the Figure 9 visualization
#################################

imgSize_in=150/25.4 #180 mm maximum divided by  in/mm ratio    
    
# row and column sharing
f, ((ax11, ax12,ax13),(ax21, ax22,ax23),(ax31, ax32,ax33),
    (ax41, ax42,ax43),(ax51, ax52,ax53),(ax61, ax62,ax63))= plt.subplots(6, 3,figsize=(imgSize_in,imgSize_in*1.7),dpi=300)

f.subplots_adjust(wspace=.4)
f.subplots_adjust(hspace=.36)


#plotting prior for boundscale parameter
x1=0
x2=0.5
x=np.linspace(x1,x2,num=100)
y=norm.pdf(x, 0, 0.1)
ax11.plot(x, y, '-',color='black',linewidth=1)	
ax11.plot([0,0], [0,3.989] , '-',color='black',linewidth=1)	
spcX=(max(x)-min(x))/25
ax11.set_xlim(min(x)-spcX,max(x)+spcX)
ax11.set_ylim(0,4.2)
ax11.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax11.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax11.set_yticks([0,2,4])
ax11.xaxis.set_tick_params(pad=-1)
ax11.yaxis.set_tick_params(pad=-1)
ax11.set_xlabel(r'$\bar{sc}$',labelpad=1,fontsize=10)
ax11.set_ylabel('pdf')
ax11.fill_between(x,y,color='silver')
ax11.get_yaxis().set_label_coords(-0.2,0.5)
ax11.get_xaxis().set_label_coords(0.5,-0.2)

ax11.annotate('A', xy=(.03, .95),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 


#plotting prior for variabilty of scale
std = 0.5
mean = -3
x=np.linspace(0,1,200)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax12.plot(x,y,color='black',linewidth=1)
spcX=(max(x)-min(x))/25
ax12.set_xlim(min(x)-spcX,max(x)+spcX)
ax12.set_ylim(0,max(y)*1.2)
ax12.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax12.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax12.set_yticks([0,10,20])
ax12.xaxis.set_tick_params(pad=-1)
ax12.yaxis.set_tick_params(pad=-1)
ax12.set_xlabel(r"$\sigma_{sc}$",labelpad=1,fontsize=10)
ax12.fill_between(x,y,color='silver')
ax12.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for bound
std = 1
mean = -3
x=np.linspace(0,0.4,200)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax13.plot(x,y,color='black',linewidth=1)
spcX=(max(x)-min(x))/25
ax13.set_xlim(min(x)-spcX,max(x)+spcX)
ax13.set_ylim(0,max(y)*1.1)
ax13.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax13.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax13.set_yticks([0,7,14])
ax13.xaxis.set_tick_params(pad=-1)
ax13.yaxis.set_tick_params(pad=-1)
ax13.set_xlabel(r"$B$",labelpad=1,fontsize=10)
ax13.fill_between(x,y,color='silver')
ax13.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for prior mean
x1=-1
x2=1
x=np.linspace(x1,x2,num=100)
y=norm.pdf(x, 0, 0.5)
ax21.plot(x, y, '-',color='black',linewidth=1)	
ax21.plot([-1,-1], [0,0.1080], '-',color='black',linewidth=1)	
ax21.plot([1,1], [0,0.1080], '-',color='black',linewidth=1)	
spcX=(max(x)-min(x))/25
ax21.set_xlim(min(x)-spcX,max(x)+spcX)
ax21.set_ylim(0,0.85)
ax21.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax21.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax21.set_yticks([0,0.4,0.8])
ax21.xaxis.set_tick_params(pad=-1)
ax21.yaxis.set_tick_params(pad=-1)
ax21.set_xlabel(r"$\bar z_0$",labelpad=1,fontsize=10)
ax21.set_ylabel('pdf')
ax21.fill_between(x,y,color='silver')
ax21.get_yaxis().set_label_coords(-0.2,0.5)
ax21.get_xaxis().set_label_coords(0.5,-0.2)



#plotting prior for prior range
std = 0.5
mean = -3.9
x=np.linspace(0,0.06,200)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax22.plot(x, y, '-',color='black',linewidth=1)	
spcX=(max(x)-min(x))/25
ax22.set_xlim(min(x)-spcX,max(x)+spcX)
ax22.set_ylim(0,50)
ax22.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
ax22.set_yticks([0,25,50])
#ax22.set_yticks([0,25,50])
ax22.xaxis.set_tick_params(pad=-1)
ax22.yaxis.set_tick_params(pad=-1)
ax22.set_xlabel(r"$s_Z$",labelpad=1,fontsize=10)
ax22.fill_between(x,y,color='silver')
ax22.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for Tnd
std = 1
mean = -2
x=np.linspace(0,1.2,200)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax23.plot(x,y,color='black',linewidth=1)
spcX=(max(x)-min(x))/25
ax23.set_xlim(min(x)-spcX,max(x)+spcX)
ax23.set_ylim(0,6)
ax23.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax23.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax23.set_yticks([0,3,6])
ax23.xaxis.set_tick_params(pad=-1)
ax23.yaxis.set_tick_params(pad=-1)
ax23.set_xlabel(r"$\bar T_{nd}$",labelpad=1,fontsize=10)
ax23.fill_between(x,y,color='silver')
ax23.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for st
std = 1
mean = -2.5
x=np.linspace(0,0.8,200)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax31.plot(x,y,color='black',linewidth=1)
spcX=(max(x)-min(x))/25
ax31.set_xlim(min(x)-spcX,max(x)+spcX)
ax31.set_ylim(0,9)
ax31.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax31.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax31.set_yticks([0,4,8])
ax31.xaxis.set_tick_params(pad=-1)
ax31.yaxis.set_tick_params(pad=-1)
ax31.set_xlabel(r"$s_t$",labelpad=1,fontsize=10)
ax31.set_ylabel('pdf')
ax31.fill_between(x,y,color='silver')
ax31.get_yaxis().set_label_coords(-0.2,0.5)
ax31.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for lapseprob
x1=0
x2=1
x=np.linspace(0,1,200)
y=gaussprobpdf(x, -1.65, 1, width=1.0, shift=0.0)
ax32.plot(x,y,color='black',linewidth=1)
spcX=(max(x)-min(x))/25
ax32.set_xlim(min(x)-spcX,max(x)+spcX)
ax32.set_ylim(0,22)
ax32.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax32.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax32.set_yticks([0,10,20])
ax32.xaxis.set_tick_params(pad=-1)
ax32.yaxis.set_tick_params(pad=-1)
ax32.set_xlabel(r"$\pi_l$",labelpad=1,fontsize=10)
ax32.fill_between(x,y,color='silver')
ax32.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for lapsetoprob
x1=0
x2=1
x=np.linspace(x1,x2,num=100)
y=gaussprobpdf(x, 0, 1, width=1.0, shift=0.0)
ax33.plot(x, y, '-',color='black',linewidth=1)	
spcX=(max(x)-min(x))/25
ax33.set_xlim(min(x)-spcX,max(x)+spcX)
ax33.set_ylim(0,max(y)*1.1)
ax33.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
ax33.set_yticks([0,0.5,1])
ax33.xaxis.set_tick_params(pad=-1)
ax33.yaxis.set_tick_params(pad=-1)
ax33.set_xlabel('$\pi_{to}$',labelpad=1,fontsize=10)
ax33.fill_between(x,y,color='silver')
ax33.get_xaxis().set_label_coords(0.5,-0.2)

######################################
##########Plot the posteriors########
######################################

mat_contents=sio.loadmat('model_fit_data\\fitResults_sub_130_cond_1_aper_12_run_5_dc_cond_final_16.mat')
posterior_mean=mat_contents['ep_mean'].ravel()
posterior_cov=mat_contents['ep_cov']
posterior_std=np.diag(posterior_cov)
 
#plotting prior for scale mean
x=np.linspace(x1,x2,num=500)
y=norm.pdf(x, posterior_mean[1], posterior_std[1])
ax41.plot(x, y, '-',color='blue',linewidth=1)	
spcX=(max(x)-min(x))/25
ax41.set_xlim(min(x)-spcX,max(x)+spcX)
ax41.set_ylim(0,max(y)*1.1)
ax41.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax41.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax41.set_yticks([0,30,60])
ax41.xaxis.set_tick_params(pad=-1)
ax41.yaxis.set_tick_params(pad=-1)
ax41.set_xlabel(r'$\bar{sc}$',labelpad=1,fontsize=10)
ax41.set_ylabel('pdf')
ax41.fill_between(x,y,color='cornflowerblue')
ax41.get_yaxis().set_label_coords(-0.2,0.5)
ax41.get_xaxis().set_label_coords(0.5,-0.2)

ax41.annotate('B', xy=(.03, .49),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 

#plotting posterior for scale std
std = posterior_std[2]
mean = posterior_mean[2]
x=np.linspace(0,1,200)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax42.plot(x,y,color='blue',linewidth=1)
spcX=(max(x)-min(x))/25
ax42.set_xlim(min(x)-spcX,max(x)+spcX)
ax42.set_ylim(0,max(y)*1.1)
ax42.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
ax42.set_yticks([0,25,50])
#ax42.set_yticks([0,0.05,0.1])
ax42.xaxis.set_tick_params(pad=-1)
ax42.yaxis.set_tick_params(pad=-1)
ax42.set_xlabel(r"$\sigma_{sc}$",labelpad=1,fontsize=10)
ax42.fill_between(x,y,color='cornflowerblue')
ax42.get_xaxis().set_label_coords(0.5,-0.2)

#plotting prior for bound
std = posterior_std[3]
mean = posterior_mean[3]
x=np.linspace(0,0.4,200)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax43.plot(x,y,color='blue',linewidth=1)
spcX=(max(x)-min(x))/25
ax43.set_xlim(min(x)-spcX,max(x)+spcX)
ax43.set_ylim(0,max(y)*1.1)
ax43.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax43.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax43.set_yticks([0,80,160])
ax43.xaxis.set_tick_params(pad=-1)
ax43.yaxis.set_tick_params(pad=-1)
ax43.set_xlabel(r"$B$",labelpad=1,fontsize=10)
ax43.fill_between(x,y,color='cornflowerblue')
ax43.get_xaxis().set_label_coords(0.5,-0.2)

#plotting prior for prior mean
x1=-1
x2=1
x=np.linspace(x1,x2,num=500)
y=norm.pdf(x, posterior_mean[4], posterior_std[4])
ax51.plot(x, y, '-',color='blue',linewidth=1)	
spcX=(max(x)-min(x))/25
ax51.set_xlim(min(x)-spcX,max(x)+spcX)
ax51.set_ylim(0,max(y)*1.1)
ax51.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax51.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax51.set_yticks([0,60,120])
ax51.xaxis.set_tick_params(pad=-1)
ax51.yaxis.set_tick_params(pad=-1)
ax51.set_xlabel(r"$\bar z_0$",labelpad=1,fontsize=10)
ax51.set_ylabel('pdf')
ax51.fill_between(x,y,color='cornflowerblue')
ax51.get_yaxis().set_label_coords(-0.2,0.5)
ax51.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for prior range
x1=0
x2=0.06
x=np.linspace(0,0.06,200)
y=lognorm.pdf(x,s=posterior_std[5],scale=np.exp(posterior_mean[5]))
ax52.plot(x, y, '-',color='blue',linewidth=1)	
spcX=(max(x)-min(x))/25
ax52.set_xlim(min(x)-spcX,max(x)+spcX)
ax52.set_ylim(0,max(y)*1.1)
ax52.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax52.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax52.set_yticks([0,70,140])
ax52.xaxis.set_tick_params(pad=-1)
ax52.yaxis.set_tick_params(pad=-1)
ax52.set_xlabel(r"$s_Z$",labelpad=1,fontsize=10)
ax52.fill_between(x,y,color='cornflowerblue')
ax52.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for Tnd
std = posterior_std[6]
mean = posterior_mean[6]
x=np.linspace(0,1.2,5000)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax53.plot(x,y,color='blue',linewidth=1)
spcX=(max(x)-min(x))/25
ax53.set_xlim(min(x)-spcX,max(x)+spcX)
ax53.set_ylim(0,max(y)*1.1)
ax53.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax53.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax53.set_yticks([0,300,600])
ax53.xaxis.set_tick_params(pad=-1)
ax53.yaxis.set_tick_params(pad=-1)
ax53.set_xlabel(r"$\bar T_{nd}$",labelpad=1,fontsize=10)
ax53.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax53.fill_between(x,y,color='cornflowerblue')
ax53.get_xaxis().set_label_coords(0.5,-0.2)

#plotting prior for st
std = posterior_std[7]
mean = posterior_mean[7]
x=np.linspace(0,0.8,500)
y=lognorm.pdf(x,s=std,scale=np.exp(mean))
ax61.plot(x,y,color='blue',linewidth=1)
spcX=(max(x)-min(x))/25
ax61.set_xlim(min(x)-spcX,max(x)+spcX)
ax61.set_ylim(0,max(y)*1.1)
ax61.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax61.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax61.set_yticks([0,50,100])
ax61.xaxis.set_tick_params(pad=-1)
ax61.yaxis.set_tick_params(pad=-1)
ax61.set_xlabel(r"$s_t$",labelpad=1,fontsize=10)
ax61.set_ylabel('pdf')
ax61.fill_between(x,y,color='cornflowerblue')
ax61.get_yaxis().set_label_coords(-0.22,0.5)
ax61.get_xaxis().set_label_coords(0.5,-0.2)


#plotting prior for lapseprob
std = posterior_std[8]
mean = posterior_mean[8]
x=np.linspace(0,1,5000)
y=gaussprobpdf(x, mean, std, width=1.0, shift=0.0)
ax62.plot(x,y,color='blue',linewidth=1)
spcX=(max(x)-min(x))/25
ax62.set_xlim(min(x)-spcX,max(x)+spcX)
ax62.set_ylim(0,max(y)*1.1)
ax62.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax62.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax62.set_yticks([0,60,120])
ax62.xaxis.set_tick_params(pad=-1)
ax62.yaxis.set_tick_params(pad=-1)
ax62.set_xlabel(r"$\pi_l$",labelpad=1,fontsize=10)
ax62.fill_between(x,y,color='cornflowerblue')
ax62.get_xaxis().set_label_coords(0.5,-0.2)

#plotting prior for lapse-TO prob
x1= 0
x2= 1
x=np.linspace(x1,x2,num=200)
y=gaussprobpdf(x, posterior_mean[9], posterior_std[9], width=1.0, shift=0.0)
ax63.plot(x, y, '-',color='blue',linewidth=1)	
spcX=(max(x)-min(x))/25
ax63.set_xlim(min(x)-spcX,max(x)+spcX)
ax63.set_ylim(0,max(y)*1.1)
ax63.set_xticks([min(x),min(x)+(max(x)-min(x))/2,max(x)])
#ax63.set_yticks([min(y),min(y)+(max(y)-min(y))/2,max(y)])
ax63.set_yticks([0,1,2])
ax63.xaxis.set_tick_params(pad=-1)
ax63.yaxis.set_tick_params(pad=-1)
ax63.set_xlabel('$\pi_{to}$',labelpad=1,fontsize=10)
ax63.fill_between(x,y,color='cornflowerblue')
ax63.get_xaxis().set_label_coords(0.5,-0.2)


#################################
#Saving the visualization to the Figures folder
#################################

savefig('figures\\Figure_9_priors_posteriors.jpg', bbox_inches='tight',dpi=300)
savefig('figures\\Figure_9_priors_posteriors.eps', bbox_inches='tight',dpi=300)
savefig('figures\\Figure_9_priors_posteriors.tiff', bbox_inches='tight',dpi=300)

