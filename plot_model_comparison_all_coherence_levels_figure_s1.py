'''
The script 'plot_model_comparison_all_coherence_levels_figure_s1.py' creates the 
visualizations related to the model comparison analysis 
for Figure S1 in "Fard et al. (2021), Spatiotemporal 
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
sns.set(style="ticks", color_codes=True,font_scale=1.1)


#################################
#loading the .mat file containing the results of the model comparison analysis
#################################

mat_contents=sio.loadmat('analysis_data//model_comparison_all_participants.mat')


#################################
#Creating the Figure S1 visualization
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio
f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(imgSize_in, imgSize_in*0.4),sharex=True,dpi=300)
plt.subplots_adjust(hspace = .28,wspace=.4) 

# plot the exceedance probability for families
EP=mat_contents['EP_2way']


EP11=np.asarray(EP[0,0])
EP12=np.asarray(EP[0,1])
EP21=np.asarray(EP[1,0])
EP22=np.asarray(EP[1,1])


ax1.bar([1.2], EP[0,0],width=0.4,color='khaki',align='center',label='DDM')
ax1.bar([1.6], EP[0,1],width=0.4,color='green',align='center',label='EXaM')

ax1.bar([2.2], EP[1,0],width=0.4,color='khaki',align='center')
ax1.bar([2.6], EP[1,1],width=0.4,color='green',align='center')

ax1.bar([3.2], EP[2,0],width=0.4,color='khaki',align='center')
ax1.bar([3.6], EP[2,1],width=0.4,color='green',align='center')

ax1.bar([4.2], EP[3,0],width=0.4,color='khaki',align='center')
ax1.bar([4.6], EP[3,1],width=0.4,color='green',align='center')


ax1.plot([0.5,5.0],[0.95,0.95],'--r')
ax1.set_ylim([0,1.2])	
ax1.set_xlim([0.8,5.0])

ax1.grid(which='both',axis='y')

ax1.set_ylabel('Exceedance probability')
ax1.set_xlabel('Coherence level')
ax1.set_xticks([1.4,2.4,3.4,4.4])

ax1.set_xticks([1.4,2.4,3.4,4.4])
ax1.set_xticklabels(['0%','10%','25%','35%'])

ax1.set_yticks([0,0.25,0.5,0.75,1])
ax1.set_yticklabels([0,0.25,0.5,0.75,1])

ax1.annotate('A', xy=(.04, .97),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 


#plot the model frequency
MF_mean=mat_contents['MF_mean_2way']
MF_std=mat_contents['MF_std_2way']


FF11=np.asarray(MF_mean[0,0])
FF12=np.asarray(MF_mean[0,1])
FF21=np.asarray(MF_mean[1,0])
FF22=np.asarray(MF_mean[1,1])
FF31=np.asarray(MF_mean[2,0])
FF32=np.asarray(MF_mean[2,1])
FF41=np.asarray(MF_mean[3,0])
FF42=np.asarray(MF_mean[3,1])

FFS11=np.asarray(MF_std[0,0])
FFS12=np.asarray(MF_std[0,1])
FFS21=np.asarray(MF_std[1,0])
FFS22=np.asarray(MF_std[1,1])
FFS31=np.asarray(MF_std[2,0])
FFS32=np.asarray(MF_std[2,1])
FFS41=np.asarray(MF_std[3,0])
FFS42=np.asarray(MF_std[3,1])

ax2.bar([1.2], FF11,width=0.4,color='khaki',align='center',label='DDM')
ax2.bar([1.6], FF12,width=0.4,color='green',align='center',label='EXaM')

ax2.bar([2.2], FF21,width=0.4,color='khaki',align='center')
ax2.bar([2.6], FF22,width=0.4,color='green',align='center')

ax2.bar([3.2], FF31,width=0.4,color='khaki',align='center')
ax2.bar([3.6], FF32,width=0.4,color='green',align='center')

ax2.bar([4.2], FF41,width=0.4,color='khaki',align='center')
ax2.bar([4.6], FF42,width=0.4,color='green',align='center')

ax2.errorbar([1.2], FF11, yerr=FFS11, fmt='r-')
ax2.errorbar([1.6], FF12, yerr=FFS12, fmt='r-')

ax2.errorbar([2.2], FF21, yerr=FFS21, fmt='r-')
ax2.errorbar([2.6], FF22, yerr=FFS22, fmt='r-')


ax2.errorbar([3.2], FF31, yerr=FFS31, fmt='r-')
ax2.errorbar([3.6], FF32, yerr=FFS32, fmt='r-')

ax2.errorbar([4.2], FF41, yerr=FFS41, fmt='r-')
ax2.errorbar([4.6], FF42, yerr=FFS42, fmt='r-')


ax2.plot([-1,5.0],[0.5,0.5],'--r')
ax2.set_ylim([0,1.2])	
ax2.set_xlim([0.8,5.0])	

ax2.set_xticks([1.4,2.4,3.4,4.4])
ax2.set_xticklabels(['0%','10%','25%','35%'])
ax2.set_yticks([0,0.25,0.5,0.75,1])
ax2.set_yticklabels([0,0.25,0.5,0.75,1])
ax2.legend(loc='upper right')
ax2.set_xlabel('Coherence level')
ax2.grid(which='both',axis='y')

ax2.legend(loc='upper right',prop={'size':9},frameon=True)#, bbox_to_anchor=(1, 0.5))
ax2.set_ylabel('Model frequency')

ax2.annotate('B', xy=(.53, .97),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 

#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_S1_model_comparison_DDM_EXaM_all_coherence_levels.jpg', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_S1_model_comparison_DDM_EXaM_all_coherence_levels.eps', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_S1_model_comparison_DDM_EXaM_all_coherence_levels.tiff', bbox_inches='tight',dpi=300)

