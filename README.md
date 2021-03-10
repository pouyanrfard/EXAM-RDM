# Exact Input Modeling of Response Consistency in Perceptual Decision Making
The code required to reproduce the results of the study "Fard et. al (2021), Spatiotemporal Modeling of Response Consistency with Incoherent Motion Stimuli in Perceptual Decision Making"

Author: Pouyan R. Fard

Email: pouyan.rafieifard@gmail.com
         
## Requirements
 - [Git](https://git-scm.com/) or another version control software
 - [Matlab R2018b](https://www.mathworks.com/products/new_products/release2018b.html) or higher version 
 - [Anaconda 3](https://www.anaconda.com/products/individual/download-success) or a distribution of Python 3.7 
 - [VBA Toolbox](https://mbb-team.github.io/VBA-toolbox/) 
 - [Matlab library for false discovery rate control procedure for a set of statistical tests](https://www.mathworks.com/matlabcentral/fileexchange/27418-fdr_bh)
 -  [Py-EPABC](https://github.com/sbitzer/pyEPABC) bt Sebastian Bitzer
 -  [rtmodel](https://github.com/sbitzer/rtmodels) by Sebastian Bitzer

## Setup
To successfuly reproduce the figures used in the publication, you have to first install the software requirements according to the checklist below:

 1. Install Git
 2. Install Matlab
 3. Install Anaconda
 4. Install the VBA Toolbox:
 
	 4.1. Download the [VBA Toolbox](https://mbb-team.github.io/VBA-toolbox/download/) 
	 
	 4.2. Unzip the downloaded zip file
	 
	 4.3. Run the VBA_setup.m via Matlab
5. Clone this Git repository in a proper folder drive via by operating the following procedure in the command line: 

`git clone https://github.com/pouyanrfard/EXAM-RDM.git`
	
6. Download the [fdr_bh.m](https://www.mathworks.com/matlabcentral/fileexchange/27418-fdr_bh) file and place it in the "functions" folder in the cloned code repository
7. **Only required if you plan to re-fit the model to the data:** Install the [Py-EPABC](https://github.com/sbitzer/pyEPABC) as a Python library using the following procedure in the command line:

`git clone https://github.com/sbitzer/pyEPABC`

`python setup.py install`

8. **Only required if you plan to re-fit the model to the data:** Install the [rtmodels](https://github.com/sbitzer/rtmodels)  as a Python library using the following procedure in the command line:

`git clone https://github.com/sbitzer/rtmodels`

`python setup.py develop`

9. **RECOMMENDED:** Install all the dependencies in Python first by navigating to the main repositoryfolder in command line and then using the following command: 

`pip3 install -r requirements.txt`

## Guidelines to reproduce the results of the paper
After successful setup of the environment you can reproduce the results of the paper (available in the "Figures" and "Tables" folders) using the following guidelines: 

**IMPORTANT NOTE: Please do not delete or move the files contained in "behavioral_data", "experiment_design", "functions", and "model_fit_data"**


**Reproduce Figure 1. B-C:** 
1. Run behavioral_analysis.m
2. Run plot_behavioral_figure_1.py

**Reproduce Figure 2.:**
1. Run behavioral_analysis.m
2. Run plot_response_consistency_figure_2.py

**Reproduce Figure 3:**
1. Run behavioral_analysis.m
2. Run plot_response_consistency_figure_2.py

**Reproduce Figure 4:** 
1. Run model_comparison_all_participants.m
2. Run plot_model_comparsion_zero_coherence_figure_4.py

**Reproduce Table 1:**
1. Run model_comparison_all_participants.m
2. Open the Excel sheet (posterior_parameter_table_0_coherence_Table_1.xlsx) in the Tables folder

**Reproduce Figure S2:** 
1. Run rank_subjects.m
2. Run plot_subject_ranking_figure_s2.py

**Reproduce Figure 5:** 
1. Run model_comparison_high_vs_low_performing_participants.m 
2. Run plot_model_comparison_high_vs_low_performing_participants_figure_5.py

**Reproduce Table 2:**
1. Run model_comparison_high_vs_low_performing_participants.m 
2. Open the Excel sheet (posterior_parameter_table_0_coherence_high_vs_low_performing.xlsx) in the Tables folder

**Reproduce Figure 6:**
1. Run regression_analysis.m
2. Run plot_regression_analysis_scale_consistency_figure_6.py

**Reproduce Figure 7:** 
1. Run behavioral_analysis.m
2. Run plot_response_distributions_figure_7.py

**Reproduce Figure 8:**
1. Run analyze_stimulus_features_coherence.m
2. Run plot_dot_counts_analysis_figure_8.py

**Reproduce Figure 9:**
1. Run plot_priors_and_posterior_examples_figure_9.py

**Reproduce Figure S1:** 
1. Run model_comparison_all_participants.m
2. Run plot_model_comparison_all_coherence_levels_figure_s1.py

**Reproduce Table S1:** 
1. Run model_comparison_all_participants.m
2. Open the Excel sheet (posterior_parameter_table_without_0_coherence_Table_S1.xlsx) in the Tables folder

**Reproduce Figure S3:**
1. Run regression_analysis.m
2. Run plot_regression_analysis_non_zero_accuracy_consistency_figure_s3.py


**Reproduce model fitting analysis for 0% coherence:**
1. Run prepareFitData_norm_dc_zeroCons.m
2. Run test_model_inference_zero_coherence.py

**Reproduce model fitting analysis for non-zero % coherence:**
1. Run test_prepare_fit_data.m
2. Run test_model_inference_non_zero_coherence.py

