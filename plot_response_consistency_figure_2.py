'''
The script 'plot_response_consistency_figure_2.py' creates the visualizations related 
the behavioral analysis for Figure 2 in "Fard et al. (2021), Spatiotemporal 
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


#%matplotlib qt
from pylab import *
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from matplotlib.ticker import AutoMinorLocator
sns.set(style="ticks", color_codes=True,font_scale=1.2)
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


#################################
#loading the .mat file containing the results of the behavioral analysis
#################################

loadFileName="analysis_data//behavioral_analysis_results.mat" 

mat_contents=sio.loadmat(loadFileName,squeeze_me=True)

consistency_map=mat_contents['frac_res'] #response consistency map
consistent_stimulus_types=mat_contents['zeroSeeds_conditions'] #index of Cluster 1 stimulus types
subjectCons=mat_contents['subjCons']-0.5 #index of participants with consistent responses
stimulusCons=mat_contents['stimulusCons']-0.5 #index of stimulus types for participants with consitent responses


#################################
#Creating the Figure 2 visualization
#################################

numSubs,numStim=consistency_map.shape
extent=(0,numStim,0,numSubs)
imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio

fig = plt.figure(figsize=(imgSize_in, imgSize_in*1.4),dpi=300)    

#################################
#Visaulizing the consistency map
#################################

ax1 = fig.add_subplot(111)

im=ax1.imshow(consistency_map, cmap='seismic',extent=extent,origin='lower',aspect='auto', interpolation='none',
              vmin=-1*np.max(consistency_map),vmax=np.max(consistency_map))
ax1.set_xlabel('Stimulus types')
ax1.set_ylabel('Participants')
ax1.set_xlim(0,numStim)    
ax1.set_ylim(0,numSubs)    
ax1.set_xticklabels('')

ax1.tick_params(
    axis='both',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off',      # ticks along the bottom edge are off
    right='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
                

# Hide major tick labels
ax1.xaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0.5, numStim+1.5, 1)))
ax1.xaxis.set_minor_formatter(ticker.FixedFormatter(['1*','2','3*','4*','5','6*','7','8','9*','10','11','12']))

# Hide major tick labels
ax1.yaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax1.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(0.5, numSubs+1.5, 1)))
ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(['1','2','3','4','5','6','7','8','9','10','11','12',
                                                     '13','14','15','16','17','18','19','20','21','22',
                                                     '23','24','25','26','27','28','29','30','31','32',
                                                     '33','34','35','36','37','38','39','40','41','42',
                                                     '43','44']))

ax1.plot(stimulusCons, subjectCons, 'o',color='yellow',markersize=8)

ax1.annotate('L', xy=(.88, .08),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=15,fontweight='bold',color='blue') 

ax1.annotate('R', xy=(.88, .952),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=15,fontweight='bold',color='red') 


cbar = plt.colorbar(im, ticks=[-0.5, -0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])


#################################
#Saving the visualization to the Figures folder
#################################

cbar.set_label('Response consistency (RC)')
plt.savefig('figures\\Figure_2_consistency_map.png', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_2_consistency_map.tif', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_2_consistency_map.eps', bbox_inches='tight',dpi=300)
