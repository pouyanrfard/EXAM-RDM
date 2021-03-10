'''
The script 'plot_subject_ranking_figure_s2.py' creates the visualizations related 
the behavioral analysis for Figure 3 in "Fard et al. (2021), Spatiotemporal 
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

def significantStar(ax,data,x1,x2,star,y,h,col):
    # statistical annotation
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col)
    

#################################
#loading the .mat file containing the results of the subject ranking
#################################

mat_contents=sio.loadmat('analysis_data//subject_ranking.mat')


#################################
#Creating the Figure S2 visualization
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio
f, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(imgSize_in, imgSize_in*0.8),sharex=True,dpi=300)
plt.subplots_adjust(hspace = .28,wspace=.32) 

# plot the exceedance probability for families
acc_res_means=mat_contents['acc_res_means']
acc_res_stds=mat_contents['acc_res_stds']


ax1.bar([1.2], acc_res_means[0,0],width=0.4,color=[255/255,20/255,147/255],align='center',label='Bottom 25%')
ax1.bar([1.6], acc_res_means[0,1],width=0.4,color=[0/255,206/255,209/255],align='center',label='Top 25%')

ax1.bar([2.2], acc_res_means[1,0],width=0.4,color=[255/255,20/255,147/255],align='center')
ax1.bar([2.6], acc_res_means[1,1],width=0.4,color=[0/255,206/255,209/255],align='center')

ax1.bar([3.2], acc_res_means[2,0],width=0.4,color=[255/255,20/255,147/255],align='center')
ax1.bar([3.6], acc_res_means[2,1],width=0.4,color=[0/255,206/255,209/255],align='center')

ax1.bar([4.2], acc_res_means[3,0],width=0.4,color=[255/255,20/255,147/255],align='center')
ax1.bar([4.6], acc_res_means[3,1],width=0.4,color=[0/255,206/255,209/255],align='center')

ax1.errorbar([1.2], acc_res_means[0,0], yerr=acc_res_stds[0,0], fmt='r-')
ax1.errorbar([1.6], acc_res_means[0,1], yerr=acc_res_stds[0,1], fmt='r-')

ax1.errorbar([2.2], acc_res_means[1,0], yerr=acc_res_stds[1,0], fmt='r-')
ax1.errorbar([2.6], acc_res_means[1,1], yerr=acc_res_stds[1,1], fmt='r-')

ax1.errorbar([3.2], acc_res_means[2,0], yerr=acc_res_stds[2,0], fmt='r-')
ax1.errorbar([3.6], acc_res_means[2,1], yerr=acc_res_stds[2,1], fmt='r-')

ax1.errorbar([4.2], acc_res_means[3,0], yerr=acc_res_stds[3,0], fmt='r-')
ax1.errorbar([4.6], acc_res_means[3,1], yerr=acc_res_stds[3,1], fmt='r-')

significantStar(ax1,acc_res_means,2.2,2.6,'***',1.07,0.02,'k')
significantStar(ax1,acc_res_means,3.2,3.6,'***',1.07,0.02,'k')
significantStar(ax1,acc_res_means,4.2,4.6,'***',1.07,0.02,'k')


ax1.set_ylim([0,1.05])	
ax1.set_xlim([0.8,5.0])


ax1.grid(which='both',axis='y')

ax1.set_ylabel('Proportion correct')
ax1.set_xticks([1.4,2.4,3.4,4.4])

ax1.set_xticks([1.4,2.4,3.4,4.4])
ax1.set_xticklabels(['0%','10%','25%','35%'])

ax1.set_ylim([0,1.3])
ax1.set_yticks([0,0.25,0.5,0.75,1])
ax1.set_yticklabels([0,'',0.5,'',1])

ax1.legend(loc='upper left',prop={'size':9},frameon=True)#, bbox_to_anchor=(1, 0.5))

ax1.annotate('A', xy=(.04, .97),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 

med_rt_res_means=mat_contents['med_rt_res_means']
med_rt_res_stds=mat_contents['med_rt_res_stds']

ax2.bar([1.2], med_rt_res_means[0,0],width=0.4,color=[255/255,20/255,147/255],align='center',label='Bottom 25%')
ax2.bar([1.6], med_rt_res_means[0,1],width=0.4,color=[0/255,206/255,209/255],align='center',label='Top 25%')

ax2.bar([2.2], med_rt_res_means[1,0],width=0.4,color=[255/255,20/255,147/255],align='center')
ax2.bar([2.6], med_rt_res_means[1,1],width=0.4,color=[0/255,206/255,209/255],align='center')

ax2.bar([3.2], med_rt_res_means[2,0],width=0.4,color=[255/255,20/255,147/255],align='center')
ax2.bar([3.6], med_rt_res_means[2,1],width=0.4,color=[0/255,206/255,209/255],align='center')

ax2.bar([4.2], med_rt_res_means[3,0],width=0.4,color=[255/255,20/255,147/255],align='center')
ax2.bar([4.6], med_rt_res_means[3,1],width=0.4,color=[0/255,206/255,209/255],align='center')

ax2.errorbar([1.2], med_rt_res_means[0,0], yerr=med_rt_res_stds[0,0], fmt='r-')
ax2.errorbar([1.6], med_rt_res_means[0,1], yerr=med_rt_res_stds[0,1], fmt='r-')

ax2.errorbar([2.2], med_rt_res_means[1,0], yerr=med_rt_res_stds[1,0], fmt='r-')
ax2.errorbar([2.6], med_rt_res_means[1,1], yerr=med_rt_res_stds[1,1], fmt='r-')

ax2.errorbar([3.2], med_rt_res_means[2,0], yerr=med_rt_res_stds[2,0], fmt='r-')
ax2.errorbar([3.6], med_rt_res_means[2,1], yerr=med_rt_res_stds[2,1], fmt='r-')

ax2.errorbar([4.2], med_rt_res_means[3,0], yerr=med_rt_res_stds[3,0], fmt='r-')
ax2.errorbar([4.6], med_rt_res_means[3,1], yerr=med_rt_res_stds[3,1], fmt='r-')


significantStar(ax2,med_rt_res_means,3.2,3.6,'**',0.95,0.04,'k')
significantStar(ax2,med_rt_res_means,4.2,4.6,'**',0.95,0.04,'k')


ax2.set_ylim([0,2])	
ax2.set_xlim([0.8,5.0])	

ax2.set_yticks([0,0.4,0.8,1.2,1.6,2])
ax2.set_xticklabels(['0%','10%','25%','35%'])
ax1.set_xticks([1.4,2.4,3.4,4.4])
ax2.set_xlabel('Coherence level')
ax2.grid(which='both',axis='y')

ax2.set_ylabel('Median RT (s)')


ax2.annotate('B', xy=(.04, .53),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 

#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_S2_subjects_ranks.jpg', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_S2_subjects_ranks.eps', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_S2_subjects_ranks.tiff', bbox_inches='tight',dpi=300)


