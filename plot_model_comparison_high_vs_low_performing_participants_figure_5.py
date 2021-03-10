'''
The script 'plot_model_comparison_high_vs_low_performing_participants_figure_5.py' creates the visualizations related 
the behavioral analysis for Figure 1 in "Fard et al. (2021), Spatiotemporal 
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
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from matplotlib.ticker import AutoMinorLocator
sns.set(style="ticks", color_codes=True,font_scale=0.9)


#################################
#loading the .mat file containing the results of the model comparison
#################################
mat_contents=sio.loadmat('analysis_data//model_comparison_high_vs_low_performing.mat')


#################################
#Creating the Figure 5.A visualization
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(imgSize_in, imgSize_in),sharex=True,dpi=300)
plt.subplots_adjust(hspace = .28,wspace=.32) 

# plot the exceedance probability for families
FEP=mat_contents['EP_2way_high']

FEP11=np.asarray(FEP[0,0])

ax1.bar([1.2], FEP[0,0],width=0.4,color='khaki',align='center',label='Average input models')
ax1.bar([1.6], FEP[0,1],width=0.4,color='green',align='center',label='Exact input models')

ax1.plot([0.5,5.0],[0.95,0.95],'--r')
ax1.set_ylim([0,1.2])	
ax1.set_xlim([0.8,2])

ax1.grid(which='both',axis='y')

ax1.set_ylabel('Exceedance probability')
ax1.set_xticks([1.4])

ax1.set_xticks([1.4])
ax1.set_xticklabels([''])

ax1.set_yticks([0,0.25,0.5,0.75,1])
ax1.set_yticklabels([0,0.25,0.5,0.75,1])

ax1.annotate('A. High-performing partticipants', xy=(.04, .97),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=16,fontweight='bold') 


#plot the model frequency
FF_mean=mat_contents['MF_mean_2way_high']
FF_std=mat_contents['MF_std_2way_high']


FF11=np.asarray(FF_mean[0,0])
FF12=np.asarray(FF_mean[0,1])

FFS11=np.asarray(FF_std[0,0])
FFS12=np.asarray(FF_std[0,1])

ax2.bar([1.2], FF11,width=0.4,color='khaki',align='center',label='DDM')
ax2.bar([1.6], FF12,width=0.4,color='green',align='center',label='EXaM')

ax2.errorbar([1.2], FF11, yerr=FFS11, fmt='r-')
ax2.errorbar([1.6], FF12, yerr=FFS12, fmt='r-')

ax2.plot([-1,5.0],[0.5,0.5],'--r')
ax2.set_ylim([0,1.2])	
ax2.set_xlim([0.8,2.0])	

ax2.set_xticks([1.4])
ax2.set_xticklabels([''])
ax2.set_yticks([0,0.25,0.5,0.75,1])
ax2.set_yticklabels([0,0.25,0.5,0.75,1])

ax2.grid(which='both',axis='y')

ax2.legend(loc='upper right',prop={'size':8})#, bbox_to_anchor=(1, 0.5))
ax2.set_ylabel('Model frequency')

#################################
#Creating the Figure 5.B visualization
#################################

# plot the exceedance probability for families
EP=mat_contents['EP_2way_low']


EP11=np.asarray(EP[0,0])


ax3.bar([1.2], EP[0,0],width=0.4,color='khaki',align='center',label='EXaM')
ax3.bar([1.6], EP[0,1],width=0.4,color='green',align='center',label='eEXaM')

ax3.plot([0.5,5.0],[0.95,0.95],'--r')
ax3.set_ylim([0,1.2])	
ax3.set_xlim([0.8,2.0])

ax3.set_xticks([1.4])
ax3.set_xticklabels(['0%'])
ax3.set_yticks([0,0.25,0.5,0.75,1])
ax3.set_yticklabels([0,0.25,0.5,0.75,1])

ax3.grid(which='both',axis='y')

ax3.set_ylabel('Exceedance probability')
ax3.set_xlabel('Coherence level')


ax3.annotate('B. Low-performing partticipants', xy=(.04, .5),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=16,fontweight='bold') 


#plot the model frequency
MF_mean=mat_contents['MF_mean_2way_low']
MF_std=mat_contents['MF_std_2way_low']


MF11=np.asarray(MF_mean[0,0])
MF12=np.asarray(MF_mean[0,1])
MFS11=np.asarray(MF_std[0,0])
MFS12=np.asarray(MF_std[0,1])

ax4.bar([1.2], MF11,width=0.4,color='khaki',align='center',label='EXaM')
ax4.bar([1.6], MF12,width=0.4,color='green',align='center',label='eEXaM')

ax4.errorbar([1.2], MF11, yerr=MFS11, fmt='r-')
ax4.errorbar([1.6], MF12, yerr=MFS12, fmt='r-')


ax4.plot([-1,5.0],[0.5,0.5],'--r')
ax4.set_ylim([0,1.2])	
ax4.set_xlim([0.8,2.0])	

ax4.set_xticks([1.4])
ax4.set_xticklabels(['0%'])
ax4.set_yticks([0,0.25,0.5,0.75,1])
ax4.set_yticklabels([0,0.25,0.5,0.75,1])


ax4.grid(which='both',axis='y')

ax4.set_ylabel('Model frequency')
ax4.set_xlabel('Coherence level')

#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_5_model_comparison_high_vs_low_performing.jpg', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_5_model_comparison_high_vs_low_performing.eps', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_5_model_comparison_high_vs_low_performing.tiff', bbox_inches='tight',dpi=300)
