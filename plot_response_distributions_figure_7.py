'''
The script 'plot_response_distributions_figure_7.py' creates the visualizations related 
the modeling response distributions for Figure 7 in "Fard et al. (2021), Spatiotemporal 
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
import matplotlib.ticker as ticker
import transforms

#################################
#loading the .mat file containing the results of the behavioral analysis
#################################

loadFileName="analysis_data//behavioral_analysis_results.mat" 
mat_contents=sio.loadmat(loadFileName,squeeze_me=True)

#################################
#Creating the Figure 7 visualization
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(imgSize_in, imgSize_in*0.8),dpi=300)
plt.subplots_adjust(hspace = .35,wspace=.45) 

# plot the exceedance probability for families
generalBias_right=mat_contents['fractions_right_res'].ravel()

ax1.bar(np.arange(1,45,1),generalBias_right,color='g',label='Response bias')
ax1.plot([0,45],[np.mean(generalBias_right),np.mean(generalBias_right)],'r--',label='Mean Value')
ax1.plot([0.5,5.0],[0.95,0.95],'--r')
ax1.set_ylim([0.3,0.7])	
ax1.set_xlim([0,45])

ax1.grid(which='major',axis='y')

ax1.legend(loc='upper right',prop={'size':9},frameon=True)
ax1.set_ylabel('Right-ward reponse bias')
ax1.set_xlabel('Participants')

ax1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 50, 5)))
ax1.xaxis.set_major_formatter(ticker.FixedFormatter(['0','5','10','15','20','25','30','35','40','45']))


ax1.annotate('A', xy=(.04, .97),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 


########################################
subBiasDist_set=mat_contents['subBiasDist_set']
bootStrapMean_set=mat_contents['bootStrapMean_set'].ravel()

nbins=17
midBin=int(nbins/2)+1

freqData=subBiasDist_set[41].ravel()

yMax= np.histogram(freqData, bins=nbins)[0][midBin]
ax2.hist(freqData,nbins,label='Frequency of R responses')
ax2.plot([bootStrapMean_set[41],bootStrapMean_set[41]],[0,yMax],'r--',label='Right-ward response bias')
ax2.plot([4],[15000],'*',color=[255/255,20/255,147/255],markersize=9,label='Observed frequencies')
ax2.plot([16],[15000],'*',color=[255/255,20/255,147/255],markersize=9)


ax2.legend(loc='upper right',prop={'size':8})#, bbox_to_anchor=(1, 0.5))
#ax2.set_xlabel('Frequency of right-ward responses')
ax2.set_ylabel('Counts')
#ax2.set_xlabel('Frequency of right-ward responses')

ax2.set_xlim([0,20])
ax2.legend(loc='upper right',prop={'size':9},frameon=True)
ax2.set_ylim([0,29000])

highTh=np.percentile(freqData, 97.5)
lowTh=np.percentile(freqData, 2.5)


ax2.annotate('B', xy=(.53, .97),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 
##############################################

freqData=subBiasDist_set[15].ravel()

yMax= np.histogram(freqData, bins=nbins)[0][midBin]

ax3.hist(freqData,nbins)

ax3.plot([bootStrapMean_set[15]+0.5,bootStrapMean_set[15]+0.5],[0,yMax],'r--')


ax3.legend(loc='upper right',prop={'size':8})#, bbox_to_anchor=(1, 0.5))
#ax2.set_xlabel('Frequency of right-ward responses')
ax3.set_ylabel('Counts')
ax3.set_xlabel('Frequency of right-ward responses')
ax3.set_ylim([0,20000])
ax3.plot([4],[15000],'*',color=[255/255,20/255,147/255],markersize=9)
ax3.plot([16],[15000],'*',color=[255/255,20/255,147/255],markersize=9)


ax3.set_xlim([0,20])

ax3.annotate('C', xy=(.04, .51),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 

#############################

freqData=subBiasDist_set[8].ravel()

yMax= np.histogram(freqData, bins=nbins)[0][midBin-1]

ax4.hist(freqData,nbins)

ax4.plot([bootStrapMean_set[8]+0.25,bootStrapMean_set[8]+0.25],[0,yMax],'r--')

ax4.set_ylim([0,20000])

ax4.legend(loc='upper right',prop={'size':8})#, bbox_to_anchor=(1, 0.5))
#ax2.set_xlabel('Frequency of right-ward responses')
#ax4.set_ylabel('Counts')
ax4.set_xlabel('Frequency of right-ward responses')

ax4.set_xlim([0,20])
ax4.plot([4],[15000],'*',color=[255/255,20/255,147/255],markersize=9)
ax4.plot([16],[15000],'*',color=[255/255,20/255,147/255],markersize=9)

ax4.annotate('D', xy=(.53, .51),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=15,fontweight='bold') 

#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_response_distributions.jpg', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_response_distributions.eps', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_response_distributions.tiff', bbox_inches='tight',dpi=300)