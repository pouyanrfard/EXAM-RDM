'''
The script 'plot_behavioral_figure_1.py' creates the visualizations related 
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

#%matplotlib qt
from pylab import *
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from PIL import Image
from matplotlib.ticker import AutoMinorLocator
sns.set(style="ticks", color_codes=True,font_scale=1.1)


#################################
#loading the .mat file containing the results of the behavioral analysis
#################################
loadFileName="analysis_data//behavioral_analysis_results.mat" 
mat_contents=sio.loadmat(loadFileName,squeeze_me=True)

accuracy_mean=mat_contents['mean_acc_res']
accruacy_se=mat_contents['se_acc_res']
medRT_mean=mat_contents['mean_med_rt_res']
medRT_se=mat_contents['se_med_rt_res']


#################################
#Creating the Figure 1 visualization
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio

fig = plt.figure(figsize=(imgSize_in, imgSize_in*1.2),dpi=300)    

#loading the picture demonstrating the experimental paradigm
paradigm= mpimg.imread('figures//experimental_paradigm.png')

gs1 = gridspec.GridSpec(1,1)

#################################
#Visaulizing the experimental paradigm
#################################

ax11 = fig.add_subplot(gs1[0],frameon=False)

imgplot = ax11.imshow(paradigm)
ax11.get_xaxis().tick_bottom()

ax11.axes.get_yaxis().set_visible(False)
ax11.axes.get_xaxis().set_visible(False)


ax11.annotate('A', xy=(.02, .965),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=18,fontweight='bold') 

gs1.tight_layout(fig, rect=[0, 0.42, 1, 1])

#################################
#Visaulizing the accuracy data accross coherence levels (Figure 1)
#################################

gs2 = gridspec.GridSpec(1, 2)
ax12 = fig.add_subplot(gs2[0])

ax12.errorbar([1], accuracy_mean[0],yerr=accruacy_se[0],fmt='bo', ecolor='r', capthick=2)
ax12.set_ylabel('Proportion of right-ward at 0%\n/correct responses ')
ax12.set_ylim(0.25,1.1) 
ax12.set_xlim(0,5)    
ax12.errorbar(np.arange(2,5,1), accuracy_mean[1:],yerr=accruacy_se[1:],fmt='-o', ecolor='r', capthick=2)
ax12.set_xlabel('Coherence Level')


# Hide major tick labels
#ax12.xaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax12.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 6, 1)))
ax12.xaxis.set_major_formatter(ticker.FixedFormatter(['','0%','10%','25%','35%','']))


# Hide major tick labels
#ax12.yaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax12.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0.25, 1.25, 0.25)))
ax12.yaxis.set_major_formatter(ticker.FixedFormatter(['0.25','0.5','0.75','1.0']))
ax12.set_yticks(np.arange(0.25, 1.25, 0.25))

ax12.grid(which='major', axis='both', linestyle=':')                


ax12.annotate('B', xy=(.02, .45),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=18,fontweight='bold') 


#################################
#Visaulizing the reaction time data accross coherence levels (Figure 1)
#################################


ax22 = fig.add_subplot(gs2[1])

ax22.errorbar(np.arange(1,5,1), medRT_mean,yerr=medRT_se,fmt='-o', ecolor='r', capthick=2)
ax22.set_xlabel('Coherence level')
ax22.set_ylabel('Median RT (s)')
ax22.set_xlim(0,5)    
ax22.set_ylim(0.4,1.6) 

# Hide major tick labels
#ax22.xaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax22.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 6, 1)))
ax22.xaxis.set_major_formatter(ticker.FixedFormatter(['','0%','10%','25%','35%','']))

# Hide major tick labels
#ax22.yaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax22.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0.4, 1.8, 0.4)))
ax22.yaxis.set_major_formatter(ticker.FixedFormatter(['0.4','0.8','1.2','1.6']))
ax22.set_yticks(np.arange(0.4, 1.8, 0.4))

gs2.tight_layout(fig, rect=[0, 0, 1, 0.42])

       
ax22.grid(which='major', axis='both', linestyle=':')                


ax22.annotate('C', xy=(.52, .45),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=18,fontweight='bold') 

left= min(gs1.left, gs2.left)
right = max(gs1.right, gs2.right)

gs1.update(left=left, right=right)
gs2.update(left=left, right=right,hspace=0.2)

#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_1_behavioral.png', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_1_behavioral.tif', bbox_inches='tight',dpi=300)
#plt.savefig('figures\\Figure_1_behavioral.eps', bbox_inches='tight',dpi=300)



