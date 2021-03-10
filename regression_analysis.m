% The script "regression_analysis.m" conducts the regression analyses 
% used in "Fard et al. (2021), Spatiotemporal Modeling of Response Consistency 
% with Incoherent Motion Stimuli in Perceptual Decision Making, In submission".
% The script first loads the analysis results from the behavioral analysis
% and ranking of the participants into high- and low-performing
% participants as well as the mean posterior parameter values from the
% model fittings both for all participants and for high- and low-performing
% participants. The script does several regression analyese as following:
% 1. Regression Analysis for Average absolute RC vs. Average non-zero 
% coherence accuracy (Figure S3)
% 2. Regression Analysis for Average posterior scale parameter over cluster 2
% stimulus types (Sc_2) vs. Average absolute RC  accross all subjects
% 3. Same regression analysis as 2 but for high-performing participants
% 4. Same regression analysis as 2 but for low-performing participants
% 5. Same regression analysis as 2 but for participants that are in between
% (neither high-performing nor low-performing participants)
%Hint: Please make sure that this script is placed next to the folder
%"model_fit_data", "behavioral_data", "analysis_data", "experiment_design", 
% and "functions". The successful running of the script is depedent on the
% data from behavioral analysis, subject ranking, model fitting, and model
% comparison results. 

%The script may take a few minutes to run depending on your system
%specifications.
%
% Author: Pouyan Rafieifard (January 2021)


clear all;
close all;
clc;
addpath('functions');

%% Loading the neccessary data-sets from the previous analyses

load('analysis_data//behavioral_analysis_results.mat','zeroSeeds_conditions','acc_res','frac_res');
load('analysis_data//subject_ranking.mat','bestSubjsNo','bestSubjs','worstSubjsNo','worstSubjs','otherSubjs');
load('analysis_data//model_comparison_high_vs_low_performing.mat','posMeansTrans_zero_low','posMeansTrans_zero_high');
load('analysis_data//model_comparison_all_participants.mat','posMeansTrans_zero');

%% Intialization and definition of the variables used in the analysis

%average abosulute RC accross all stimulus types
average_absolute_rc=mean(abs(frac_res),2);
%average abosulute RC accross Clusetr 2 stimulus types
sel_stimulus_types=find(zeroSeeds_conditions==1);
frac_res_sel=frac_res(:,sel_stimulus_types);
frac_res_sel_mean=mean(abs(frac_res_sel),2);
average_abosolute_rc_stimulus_type_2=frac_res_sel_mean;
%Index of top 25% participants
bestSubjs;
%Index of bottom 25% participants
worstSubjs;
%Index of other participants
otherSubjs;
% Non-zero coherence accuracy of all participants
acc_res_total=(acc_res(:,2)*240+acc_res(:,3)*160+acc_res(:,4)*160)/640;
% posterior scale parameter for zero coherence
scale_posterior_zero=squeeze(posMeansTrans_zero(:,2,2));


%% Regression Analysis for Average absolute RC vs. Average non-zero coherence accuracy

%fitting a linear regression model to the data y~ p0+p1*x
mdl_acc_cons=fitlm(acc_res_total,average_absolute_rc)

%p-value from the regression analysis
pVals_cons=mdl_acc_cons.Coefficients.pValue(2);
%the p0 parameter from the regression analysis
p0_cons=mdl_acc_cons.Coefficients.Estimate(1);
%the p1 parameter from the regression analysis
p1_cons=mdl_acc_cons.Coefficients.Estimate(2);
%the adjusted r-squared parameter from the regression analysis
rsq_adj_cons=mdl_acc_cons.Rsquared.Adjusted;

%compute the correlation coefficient between two variables
R_cons_acc = corrcoef(acc_res_total,average_absolute_rc);
R_cons_acc=R_cons_acc(1,2);

%visualize the results of the regression analysis
figure, hold on
plot(acc_res_total,average_absolute_rc,'o');
x=min(acc_res_total)-0.1:0.01:max(acc_res_total)+0.1;
plot(x,p1_cons*x+p0_cons,'k--');
xlabel('Non-zero coherence accuracy')
ylabel('Average absolute RC')
title(strcat('Regression analysis accross all Participants, R=',num2str(R_cons_acc,2),' P-val=',num2str(pVals_cons,1)))


%%  Regression Analysis for Average posterior scale vs. Average absolute RC  accross all subjects

%removing the participants with high scale parameters (outliers)
average_abosolute_rc_stimulus_type_2_outlier_removed=average_abosolute_rc_stimulus_type_2(scale_posterior_zero<0.1);
scale_posterior_zero_outlier_removed=scale_posterior_zero(scale_posterior_zero<0.1);

%fitting a linear regression model to the data y~ p0+p1*x
mdl_acc_scale=fitlm(average_abosolute_rc_stimulus_type_2_outlier_removed,scale_posterior_zero_outlier_removed)

%p-value from the regression analysis
pVals_scale=mdl_acc_scale.Coefficients.pValue(2);
%the p0 parameter from the regression analysis
p0_scale=mdl_acc_scale.Coefficients.Estimate(1);
%the p1 parameter from the regression analysis
p1_scale=mdl_acc_scale.Coefficients.Estimate(2);
%the adjusted r-squared parameter from the regression analysis
rsq_adj_scale=mdl_acc_scale.Rsquared.Adjusted;

%compute the correlation coefficient between two variables
R_scale_cons_all = corrcoef(scale_posterior_zero_outlier_removed,average_abosolute_rc_stimulus_type_2_outlier_removed);
R_scale_cons_all=R_scale_cons_all(1,2);

%visualize the results of the regression analysis
figure, hold on
plot(average_abosolute_rc_stimulus_type_2_outlier_removed,scale_posterior_zero_outlier_removed,'o');
x=min(average_abosolute_rc_stimulus_type_2_outlier_removed)-0.1:0.01:max(average_abosolute_rc_stimulus_type_2_outlier_removed)+0.1;
plot(x,p1_scale*x+p0_scale,'k--');
ylabel('Average posterior scale for Cluster 2 stimulus types')
xlabel('Average absolute RC accross Cluster 2 stimulus types')
title(strcat('Regression analysis accross all Participants, R=',num2str(R_scale_cons_all,2),' P-val=',num2str(pVals_scale,1)))


%%  Regression Analysis for Average posterior scale vs. Average absolute RC  accross top 25% subjects (high-performing participants)

%selecting only high-performing participants
scale_posterior_zero_best=scale_posterior_zero(bestSubjs);
average_abosolute_rc_stimulus_type_2_best=average_abosolute_rc_stimulus_type_2(bestSubjs);

%removing the participants with high scale parameters (outliers)
average_abosolute_rc_stimulus_type_2_best=average_abosolute_rc_stimulus_type_2_best(scale_posterior_zero_best<0.1);
scale_posterior_zero_best=scale_posterior_zero_best(scale_posterior_zero_best<0.1);

%fitting a linear regression model to the data y~ p0+p1*x
mdl_acc_scale_best=fitlm(average_abosolute_rc_stimulus_type_2_best,scale_posterior_zero_best)

%p-value from the regression analysis
pVals_scale_best=mdl_acc_scale_best.Coefficients.pValue(2);
%the p0 parameter from the regression analysis
p0_scale_best=mdl_acc_scale_best.Coefficients.Estimate(1);
%the p1 parameter from the regression analysis
p1_scale_best=mdl_acc_scale_best.Coefficients.Estimate(2);
%the adjusted r-squared parameter from the regression analysis
rsq_adj_scale_best=mdl_acc_scale_best.Rsquared.Adjusted;

%compute the correlation coefficient between two variables
R_scale_cons_best = corrcoef(average_abosolute_rc_stimulus_type_2_best,scale_posterior_zero_best);
R_scale_cons_best=R_scale_cons_best(1,2);

%visualize the results of the regression analysis
figure, hold on
plot(average_abosolute_rc_stimulus_type_2_best,scale_posterior_zero_best,'o');
x=min(average_abosolute_rc_stimulus_type_2_best)-0.1:0.01:max(average_abosolute_rc_stimulus_type_2_best)+0.1;
plot(x,p1_scale_best*x+p0_scale_best,'k--');
ylabel('Average posterior scale for Cluster 2 stimulus types')
xlabel('Average absolute RC accross Cluster 2 stimulus types')
title(strcat('Regression analysis accross top 25% Participants, R=',num2str(R_scale_cons_best,2),' P-val=',num2str(pVals_scale_best,1)))


%%  Regression Analysis for Average posterior scale vs. Average absolute RC  accross bottom 25% subjects (low-performing participants)

%selecting only low-perofrming participants
scale_posterior_zero_worst=scale_posterior_zero(worstSubjs);
average_abosolute_rc_stimulus_type_2_worst=average_abosolute_rc_stimulus_type_2(worstSubjs);


%fitting a linear regression model to the data y~ p0+p1*x
mdl_acc_scale_worst=fitlm(average_abosolute_rc_stimulus_type_2_worst,scale_posterior_zero_worst)

%p-value from the regression analysis
pVals_scale_worst=mdl_acc_scale_worst.Coefficients.pValue(2);
%the p0 parameter from the regression analysis
p0_scale_worst=mdl_acc_scale_worst.Coefficients.Estimate(1);
%the p1 parameter from the regression analysis
p1_scale_worst=mdl_acc_scale_worst.Coefficients.Estimate(2);
%the adjusted r-squared parameter from the regression analysis
rsq_adj_scale_worst=mdl_acc_scale_worst.Rsquared.Adjusted;

%compute the correlation coefficient between two variables
R_scale_cons_worst = corrcoef(average_abosolute_rc_stimulus_type_2_worst,scale_posterior_zero_worst);
R_scale_cons_worst=R_scale_cons_worst(1,2);

%visualize the results of the regression analysis
figure, hold on
plot(average_abosolute_rc_stimulus_type_2_worst,scale_posterior_zero_worst,'o');
x=min(average_abosolute_rc_stimulus_type_2_worst)-0.1:0.01:max(average_abosolute_rc_stimulus_type_2_worst)+0.1;
plot(x,p1_scale_worst*x+p0_scale_worst,'k--');
ylabel('Average posterior scale for Cluster 2 stimulus types')
xlabel('Average absolute RC accross Cluster 2 stimulus types')
title(strcat('Regression analysis accross bottom 25% Participants, R=',num2str(R_scale_cons_worst,2),' P-val=',num2str(pVals_scale_worst,1)))


%%  Regression Analysis for Average posterior scale vs. Average absolute RC  for other participants

%selecting only other participants (every participants except for high- and
%low-performing participants)
scale_posterior_zero_other=scale_posterior_zero(otherSubjs);
average_abosolute_rc_stimulus_type_2_other=average_abosolute_rc_stimulus_type_2(otherSubjs);

%fitting a linear regression model to the data y~ p0+p1*x
mdl_acc_scale_other=fitlm(average_abosolute_rc_stimulus_type_2_other,scale_posterior_zero_other);

%p-value from the regression analysis
pVals_scale_other=mdl_acc_scale_other.Coefficients.pValue(2);
%the p0 parameter from the regression analysis
p0_scale_other=mdl_acc_scale_other.Coefficients.Estimate(1);
%the p1 parameter from the regression analysis
p1_scale_other=mdl_acc_scale_other.Coefficients.Estimate(2);
%the adjusted r-squared parameter from the regression analysis
rsq_adj_scale_other=mdl_acc_scale_other.Rsquared.Adjusted;

%compute the correlation coefficient between two variables
R_scale_cons_other = corrcoef(average_abosolute_rc_stimulus_type_2_other,scale_posterior_zero_other);
R_scale_cons_other=R_scale_cons_other(1,2);

%visualize the results of the regression analysis
figure, hold on
plot(average_abosolute_rc_stimulus_type_2_other,scale_posterior_zero_other,'o');
x=min(average_abosolute_rc_stimulus_type_2_other)-0.1:0.01:max(average_abosolute_rc_stimulus_type_2_other)+0.1;
plot(x,p1_scale_other*x+p0_scale_other,'k--');
ylabel('Average posterior scale for Cluster 2 stimulus types')
xlabel('Average absolute RC accross Cluster 2 stimulus types')
title(strcat('Regression analysis accross other Participants, R=',num2str(R_scale_cons_other,2),' P-val=',num2str(pVals_scale_other,1)))

%% Save the analysis data
save('analysis_data\\regression_analysis.mat');