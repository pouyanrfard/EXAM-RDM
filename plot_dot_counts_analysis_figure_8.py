'''
The script 'plot_dot_counts_analysis_figure_8.py' creates the
visualizations related the dot counts analysis for Figure 8 in "Fard et al. (2021), 
Spatiotemporal Modeling of Response Consistency with Incoherent Motion Stimuli in 
Perceptual Decision Making, In submission".

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
sns.set(style="ticks", color_codes=True,font_scale=1.1)
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.image as mpimg

#################################
#loading the picture file for the figure
#################################

dotCounts= mpimg.imread('figures\\dotCounts.png')

#################################
#Creating the Figure 8.A visualization
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio

fig = plt.figure(figsize=(imgSize_in, imgSize_in*1.1),dpi=300)    

gs1 = gridspec.GridSpec(1, 1)
ax11 = fig.add_subplot(gs1[0],frameon=False)

imgplot = ax11.imshow(dotCounts)
ax11.get_xaxis().tick_bottom()
ax11.axes.get_yaxis().set_visible(False)
ax11.axes.get_xaxis().set_visible(False)

ax11.annotate('A', xy=(.02, .965),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=18,fontweight='bold') 

gs1.tight_layout(fig, rect=[0.05, 0.45, 1, 1])



#################################
#loading the .mat file for the dot counts analysis
#################################

mat_contents=sio.loadmat("analysis_data\\dot_counts_analysis_results.mat" )

features_tmp=mat_contents['zeroDotFeatures_dc'].ravel()
exact_features=features_tmp
dirMeans=mat_contents['dirMeans'].ravel()
dirSEs=mat_contents['dirSEs'].ravel()

#################################
#Creating the Figure 8.B visualization
#################################

gs2 = gridspec.GridSpec(1, 2)
ax21 = fig.add_subplot(gs2[0])
ax21.plot(exact_features,'k-',linewidth=1)
ax21.plot([0,121],[0,0],'--',color='gray',linewidth=0.5)

xTicks = [0,31,61,91,121]
xLabels = ['0','0.5','1','1.5','2']
T=len(exact_features)

maxPlot=np.max(exact_features)*1.2
minPlot=np.min(exact_features)*1.2

ax21.set_ylabel('Dot counts (arb. unit)')
ax21.set_xlabel('Time (s)')
ax21.set_xticks(xTicks)
ax21.set_xticklabels(xLabels)
ax21.set_ylim(minPlot,maxPlot)
ax21.set_xlim(0,T)
#ax21.grid(which='major', axis='x', linestyle=':')

ax21.annotate('R', xy=(.14, .410),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=15,fontweight='bold',color='red') 


ax21.annotate('L', xy=(.14, .110),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=15,fontweight='bold',color='blue') 


ax21.annotate('B', xy=(.02, .46),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=18,fontweight='bold') 

#################################
#Creating the Figure 8.C visualization
#################################

ax22 = fig.add_subplot(gs2[1])
ax22.plot([0,5], [0,0], 'k-')

ax22.bar([1], dirMeans[0],width=0.3,color='blue',align='center')
ax22.errorbar([1], dirMeans[0], yerr=dirSEs[0], fmt='r.')

ax22.bar(np.arange(1.85,4.85,1), dirMeans[1:4],width=0.3,color='red',align='center',label='Right')
ax22.errorbar(np.arange(1.85,4.85,1), dirMeans[1:4], yerr=dirSEs[1:4], fmt='r.')

ax22.bar(np.arange(2.15,4.15,1), dirMeans[4:7],width=0.3,color='blue',align='center',label='Left')
ax22.errorbar(np.arange(2.15,4.15,1), dirMeans[4:7], yerr=dirSEs[4:7], fmt='r.')


#ax22.xaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax22.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,6)))
ax22.xaxis.set_major_formatter(ticker.FixedFormatter(['','0%','10%','25%','35%','']))
ax22.set_xlim([0,5])

ax22.set_ylim([-3,3])


ax22.set_xlabel('Coherence level')
ax22.set_ylabel('Average normalized dot counts')

ax21.annotate('C', xy=(.51, .46),xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=18,fontweight='bold') 
ax22.legend(loc='lower left', frameon=True)

gs2.tight_layout(fig, rect=[0, 0, 1, 0.45])

#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\plot_dotCounts.png', bbox_inches='tight',dpi=300)
plt.savefig('figures\\plot_dotCounts.tif', bbox_inches='tight',dpi=300)
#plt.savefig('figures\\plot_dotCounts.eps', bbox_inches='tight',dpi=300)



