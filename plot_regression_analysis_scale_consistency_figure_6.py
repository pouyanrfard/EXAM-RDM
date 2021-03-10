'''
The script 'plot_regression_analysis_scale_consistency_figure_6' creates the
visualizations related the regression analysis for Figure 1 in "Fard et al. (2021), 
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
#Creating the Figure 7 visualization
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio

fig = plt.figure(figsize=(imgSize_in, imgSize_in*0.8),dpi=300)    


scale=mat_contents['scale_posterior_zero_outlier_removed'].ravel()
p0_scale=mat_contents['p0_scale']
p1_scale=mat_contents['p1_scale']
cons_sel_mean=mat_contents['average_abosolute_rc_stimulus_type_2_outlier_removed']
r_squared_scale=mat_contents['rsq_adj_scale']

scale_posterior_zero_best=mat_contents['scale_posterior_zero_best'].ravel()
scale_posterior_zero_worst=mat_contents['scale_posterior_zero_worst'].ravel()
scale_posterior_zero_other=mat_contents['scale_posterior_zero_other'].ravel()

frac_res_sel_mean_best=mat_contents['average_abosolute_rc_stimulus_type_2_best']
frac_res_sel_mean_worst=mat_contents['average_abosolute_rc_stimulus_type_2_worst']
frac_res_sel_mean_other=mat_contents['average_abosolute_rc_stimulus_type_2_other']

correlation_scale_cons=mat_contents['R_scale_cons_all']

plt.plot(frac_res_sel_mean_best,scale_posterior_zero_best,'o',color=	[255/255,20/255,147/255],markersize=4,label='High-performing Participants')
plt.plot(frac_res_sel_mean_worst,scale_posterior_zero_worst,'o',color=	[0/255,206/255,209/255],markersize=4,label='Low-performing Participants')
plt.plot(frac_res_sel_mean_other,scale_posterior_zero_other,'o',color=[128/255,128/255,128/255],markersize=4,label='Other Participants')

x=np.arange(min(cons_sel_mean)-0.1,max(cons_sel_mean)+0.07,0.01)
y=p1_scale*x+p0_scale

plt.plot(x,y,'k-',label='Regression \nLine')#,color=[82/255,139/255,139/255])
plt.xlabel('Average absolute RC \n across stimulus types Cluster 2')
plt.ylabel('Average posterior scale \n for stimulus types Cluster 2')


plt.xlim(0.05,0.35)
plt.xticks(np.arange(0.05,0.4,0.1))
plt.yticks(np.arange(0.0,0.09,0.02))

rsquared_text='R='+str(round(correlation_scale_cons,2))

plt.annotate(rsquared_text, xy=(.13, .92),xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=13,fontweight='bold') 
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_6_regression_analysis_scale_RC.png', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_6_regression_analysis_scale_RC.tif', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_6_regression_analysis_scale_RC.eps', bbox_inches='tight',dpi=300)


