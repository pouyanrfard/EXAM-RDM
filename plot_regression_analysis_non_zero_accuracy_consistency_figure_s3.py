'''
The script 'plot_regression_analysis_non_zero_accuracy_consistency_figure_s3' creates the
visualizations related the regression analysis for Figure S3 in "Fard et al. (2021), 
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
sns.set(style="ticks", color_codes=True,font_scale=1.2)
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.image as mpimg


#################################
#loading the .mat file containing the results of the regression analysis
#################################

loadFileName="analysis_data\\regression_analysis.mat" 
mat_contents=sio.loadmat(loadFileName,squeeze_me=True)


#################################
#Creating the Figure S3 visualization
#################################


imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio
fig = plt.figure(figsize=(imgSize_in*0.5, imgSize_in*0.5),dpi=300)  

acc_total=mat_contents['acc_res_total'].ravel()
cons_mean=mat_contents['average_absolute_rc'].ravel()
p0_cons=mat_contents['p0_cons']
p1_cons=mat_contents['p1_cons']
r_squared_cons=mat_contents['rsq_adj_cons']

correlation_cons_pc=mat_contents['R_cons_acc']

plt.plot(acc_total,cons_mean, 'ro',color='darkblue',markersize=4,label='All stimulus types')

x=np.arange(min(acc_total)-0.05,max(acc_total)+0.1,0.01)

y=p1_cons*x+p0_cons
	
plt.plot(x,y,'k-',label='Regression \nLine')#,color=[205/255,16/255,118/255])
plt.xlabel('Proportion correct at \n non-zero coherence levels')
plt.ylabel('Average absolute RC \n across all stimulus types')
plt.xlim(0.6,0.9)
plt.ylim(0.05,0.25)

plt.xticks(np.arange(0.6,1,0.1))
#plt.xticklabels(['0.6','0.7','0.8','0.9'])

plt.yticks(np.arange(0.05,0.3,0.05))

rsquared_text='R='+str(round(correlation_cons_pc,2))

plt.annotate(rsquared_text, xy=(.27, .92),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=13,fontweight='bold') 


#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_S3_regression_analysis.png', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_S3_regression_analysis.tif', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_S3_regression_analysis.eps', bbox_inches='tight',dpi=300)

