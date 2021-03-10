'''
The script 'plot_stimulus_type_clustering_figure_3.py' creates the visualizations related 
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
from sklearn.cluster import KMeans


#################################
#loading the .mat file containing the results of the behavioral analysis
#################################

loadFileName="analysis_data//stimulus_types_ranking.mat" 

mat_contents=sio.loadmat(loadFileName,squeeze_me=True)

meanCons=mat_contents['frac_res_sorted']
cons_idx=mat_contents['sorted_idx'].ravel()

#################################
#conducting the k-means clustering of the stimulus types based on their average response consistency
#################################

X = np.array([[meanCons[0]],
              [meanCons[1]],
              [meanCons[2]],
              [meanCons[3]],
              [meanCons[4]],
              [meanCons[5]],
              [meanCons[6]],
              [meanCons[7]],
              [meanCons[8]],
              [meanCons[9]],
              [meanCons[10]],
              [meanCons[11]]])

meanCons_mean=np.mean(meanCons)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)


#################################
#Visaulizing the results of clustering analysis
#################################

imgSize_in=180/25.4 #180 mm maximum divided by  in/mm ratio

fig = plt.figure(figsize=(imgSize_in, imgSize_in*0.7),dpi=300)    
gs1 = gridspec.GridSpec(1, 1)
ax11 = fig.add_subplot(gs1[0])
colors = ['orangered','darkcyan']
labelTexts=['Cluster 2','Cluster 1']

c0=0
c1=0

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    if(c0==0 and labels[i]==0):
        ax11.plot(i+1, X[i], '.',color=colors[labels[i]], markersize = 15,label=labelTexts[labels[i]])
        c0=c0+1
    elif(c1==0 and labels[i]==1):
        ax11.plot(i+1, X[i], '.',color=colors[labels[i]], markersize = 15,label=labelTexts[labels[i]])
        c1=c1+1
    else:
        ax11.plot(i+1, X[i], '.',color=colors[labels[i]], markersize = 15)

    
    
ax11.plot([0,13],[meanCons_mean,meanCons_mean],'r--',label='Mean Value')
    
plt.rc('grid', linestyle="-", color='black')
ax11.grid(True,axis='y')

ax11.set_xlim([0,13])
ax11.xaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax11.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,14)))
ax11.xaxis.set_major_formatter(ticker.FixedFormatter(['',cons_idx[0],cons_idx[1],cons_idx[2],cons_idx[3],
                                                      cons_idx[4],cons_idx[5],cons_idx[6],cons_idx[7],
                                                      cons_idx[8],cons_idx[9],cons_idx[10],cons_idx[11],'']))
ax11.set_xlabel('Stimulus types')
ax11.set_ylabel('Absolute average RC')
ax11.legend(loc='upper right',frameon=True)

#ax11.xaxis.set_major_locator(ticker.FixedLocator([0]))
#ax11.xaxis.set_major_formatter(ticker.FixedFormatter(['','']))

handles, labels = ax11.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax11.legend(handles, labels,frameon=True)


#################################
#Saving the visualization to the Figures folder
#################################

plt.savefig('figures\\Figure_3_clustering.png', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_3_clustering.tif', bbox_inches='tight',dpi=300)
plt.savefig('figures\\Figure_3_clustering.eps', bbox_inches='tight',dpi=300)



